import numpy as np
import cv2
import nd2
import os
import multipagetiff as mtif
from xml.etree import ElementTree as ET
from readlif.reader import LifFile
from multiprocessing import Pool, cpu_count
import time

from InfoIO import get_img_xml_info, get_img_nd2_info, get_img_lif_info, get_img_txt_info, save_img_xml_info
from FileRename import rename_file, rename_file_mtif
from TileCont import judge_tile_cont
from ImgIO import get_img_from_nd2, get_img_from_lif, get_img_from_mtif, import_img_one_tile, export_img_hori_stit
from ParaEsti import pyr_down_time_esti
from ImgProcess import pyr_down_img
from ImgBorder import get_2img_border, get_border_pyr_down
from ImgOvl import get_ovl_img
from LossFunc import loss_func_for_list
from AxisRange import calc_axis_range, find_first_last_index
from FFTPeak import find_peak, get_all_possible_shift


def imshow2(img1, img2, wait_time=0):
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()


def get_stitch_result(res_list, tile_num, voxel_len):
    tile_shift_arr = np.zeros((tile_num, tile_num, 3), dtype='float64')
    tile_shift_loss = -2 * np.ones((tile_num, tile_num), dtype='float64')
    num = 0
    for k in res_list:
        i, j = k[0], k[1]
        res = k[2].get()
        tile_shift_arr[i, j, :] = np.array(res[0], dtype='float64') * voxel_len
        tile_shift_arr[j, i, :] = -tile_shift_arr[i, j, :]
        tile_shift_loss[i, j] = res[1]
        tile_shift_loss[j, i] = tile_shift_loss[i, j]
        num = num+1
    return tile_shift_arr, tile_shift_loss


def update_strip_pos_by_MST(tile_pos, tile_shift_arr, tile_shift_loss):
    tile_num = tile_pos.shape[0]
    if_tile_stitched = np.zeros(tile_num, dtype='bool')
    if_tile_stitched[0] = True
    tile_refer_id = -np.ones(tile_num, dtype='int64')
    tile_shift_loss_minus = 1-tile_shift_loss
    while False in if_tile_stitched:
        loss_min = np.inf
        stitch_id = -1
        refer_id = -1
        for i in range(tile_num):
            if if_tile_stitched[i]:
                continue
            for j in range(tile_num):
                if not if_tile_stitched[j]:
                    continue
                if tile_shift_loss_minus[i, j] < loss_min:
                    loss_min = tile_shift_loss_minus[i, j]
                    stitch_id = i
                    refer_id = j
        if stitch_id == -1:
            break
        if_tile_stitched[stitch_id] = True
        tile_refer_id[stitch_id] = refer_id
    tile_refer_id_saved = tile_refer_id.copy()

    def change_with_child(father_id, shift_vec):
        tile_pos_update[father_id, :] = tile_pos_update[father_id, :] + shift_vec
        for [child_id] in np.argwhere(tile_refer_id == father_id):
            change_with_child(child_id, shift_vec)

    tile_pos_update = tile_pos.copy()
    while np.any(tile_refer_id != -1):
        for i in range(tile_num):
            for [j] in np.argwhere(tile_refer_id == -1):
                for [k] in np.argwhere(tile_refer_id == j):
                    change_with_child(k, tile_shift_arr[j, k, :])
                    tile_refer_id[k] = -1
    return tile_pos_update, tile_refer_id_saved


def one_stitch(img1, img2, tile_pos, dim_elem_num, dim_len, voxel_len, voxel_range, if_sparce,
               pyr_down_times, blur_kernel_size):
    # MIP
    img1 = cv2.medianBlur(img1, blur_kernel_size)
    img2 = cv2.medianBlur(img2, blur_kernel_size)
    for pdt in range(pyr_down_times, -1, -1):
        if pdt == pyr_down_times:
            x_s, y_s, z_s = 0, 0, 0
            x_pd, y_pd, z_pd = np.int64(np.round(np.array(voxel_range, dtype='int64') / (2 ** pdt)))
        else:
            x_s, y_s, z_s = 2 * np.array((x_s, y_s, z_s), dtype='int64')
            x_pd = y_pd = z_pd = 2 + pdt
            z_pd = 0
        loss_max = -1
        if pdt != 0:
            img1_pd = pyr_down_img(img1, pdt)
            img2_pd = pyr_down_img(img2, pdt)
            down_multi = np.array([img1_pd.shape[1], img1_pd.shape[0], img1_pd.shape[2]], dtype='float32') / np.array(
                [img1.shape[1], img1.shape[0], img1.shape[2]], dtype='float32')
        else:
            img1_pd = img1.copy()
            img2_pd = img2.copy()
            down_multi = np.array([1, 1, 1], dtype='float32')
        x_sr, y_sr, z_sr = x_s, y_s, z_s
        for x in range(x_sr - x_pd, x_sr + x_pd + 1):
            for y in range(y_sr - y_pd, y_sr + y_pd + 1):
                for z in range(z_sr - z_pd, z_sr + z_pd + 1):
                    this_tile_pos = tile_pos.copy()
                    this_tile_pos[1, :] = this_tile_pos[1, :] + voxel_len * np.array([x, y, z], dtype='float64') / down_multi
                    border = get_2img_border(dim_elem_num, dim_len, voxel_len, this_tile_pos)
                    if pdt != 0:
                        border = get_border_pyr_down(border, down_multi)
                    if 0 in border.shape:
                        continue
                    ovl1_list, ovl2_list = get_ovl_img(img1_pd, img2_pd, border, if_sparce)
                    this_loss = loss_func_for_list(ovl1_list, ovl2_list)
                    if this_loss > loss_max and this_loss > 0.5:
                        loss_max = this_loss
                        x_s, y_s, z_s = x, y, z
        print(pdt, 'down times,', 'shift is ', [x_s, y_s, z_s], 'ncc is ', loss_max)
    if loss_max <= 0.6:
        x_s, y_s, z_s = 0, 0, 0
        loss_max = -1
    if np.abs(x_s) > np.abs(voxel_range[0]) or np.abs(y_s) > np.abs(voxel_range[1]) or np.abs(z_s) > np.abs(voxel_range[2]):
        x_s, y_s, z_s = 0, 0, 0
        loss_max = -1
    print(0, 'down times,', 'shift is ', [x_s*2, y_s*2, z_s*2], 'ncc is ', loss_max)
    return [x_s*2, y_s*2, z_s*2], loss_max


def start_multi_stitch(info_IO_path, info_file_path, img_file_type, img_path, img_save_path, img_name_format,
                       img_name, ch_num, ch_th, img_data_type, if_pos_info, move_ratio, if_sparce, if_high_noise,
                       if_rename_file, if_blend, pro_num):
    # info input
    if img_file_type == 'nd2':
        whole_img = nd2.ND2File(img_path)
        dim_elem_num, dim_len, voxel_len, tile_num, tile_pos, dim_num, img_data_type = get_img_nd2_info(whole_img)
        whole_img.close()
        whole_img = nd2.imread(img_path)
    elif img_file_type == 'lif':
        whole_img = LifFile(img_path)
        dim_elem_num, dim_len, voxel_len, tile_num, tile_pos = get_img_lif_info(whole_img)
        whole_img = whole_img.get_image(0)
    elif img_file_type == 'mtif':
        dim_elem_num, dim_len, voxel_len, tile_num, tile_pos = get_img_xml_info(info_file_path)
        if if_rename_file:
            rename_file_mtif(img_name_format, img_path, img_name, ch_num)
        whole_img = np.array(mtif.read_stack(os.path.join(img_path, img_name_format % (img_name, ch_th, 'tif'))))
    elif img_file_type == 'tif':
        whole_img = None
        # dim_elem_num, dim_len, voxel_len, tile_num, tile_pos = get_img_xml_info(info_file_path)
        dim_elem_num, dim_len, voxel_len, tile_num, tile_pos = get_img_txt_info(img_path, info_file_path, ch_num)
        if if_rename_file:
            rename_file(img_name_format, img_path, img_name, dim_elem_num[2], ch_num, img_file_type)
        if img_data_type == 'uint8':
            img_mode = cv2.IMREAD_GRAYSCALE
        elif img_data_type == 'uint16':
            img_mode = cv2.IMREAD_UNCHANGED
        else:
            print('Input Error')
            return
    else:
        print('Input Error')
        return

    # tile contact
    tile_contact = judge_tile_cont(dim_len, tile_pos)
    print('contact info has been calculated.')

    # info prepare
    voxel_range = np.int64(np.ceil(dim_elem_num * move_ratio))
    print('move ratio:', voxel_range)
    pyr_down_times = pyr_down_time_esti(dim_elem_num, v_thred=100 * 100 * 20)
    print('down sample times:', pyr_down_times)
    if if_high_noise:
        blur_kernel_size = 5
    else:
        blur_kernel_size = 3

    # start multi stitch
    if pro_num == -1:
        pro_num = int(np.ceil(0.5*cpu_count()))
    print('process num:', pro_num)
    pool = Pool(processes=pro_num)
    res_list = []
    for i in range(tile_num):
        for j in range(tile_num):
            if i >= j:
                continue
            if tile_contact[i, j]:
                if img_file_type == 'nd2':
                    img1 = get_img_from_nd2(whole_img, i, ch_num, ch_th)
                    img2 = get_img_from_nd2(whole_img, j, ch_num, ch_th)
                elif img_file_type == 'lif':
                    img1 = get_img_from_lif(whole_img, i, ch_th, dim_elem_num)
                    img2 = get_img_from_lif(whole_img, j, ch_th, dim_elem_num)
                elif img_file_type == 'mtif':
                    img1 = get_img_from_mtif(whole_img, i, dim_elem_num)
                    img2 = get_img_from_mtif(whole_img, j, dim_elem_num)
                elif img_file_type == 'tif':
                    img1 = import_img_one_tile(img_name_format, img_path, img_name, i, ch_th, dim_elem_num,
                                               img_type=img_file_type, img_data_type=img_data_type, img_mode=img_mode)
                    img2 = import_img_one_tile(img_name_format, img_path, img_name, j, ch_th, dim_elem_num,
                                               img_type=img_file_type, img_data_type=img_data_type, img_mode=img_mode)
                res_list.append([i, j, pool.apply_async(one_stitch, args=(
                    img1, img2, tile_pos[[i, j], :], dim_elem_num, dim_len, voxel_len, voxel_range,
                    if_sparce, pyr_down_times, blur_kernel_size))])
    pool.close()
    pool.join()

    # update pos
    # tile_shift_arr[i, j], j相对于i的位移值
    tile_shift_arr, tile_shift_loss = get_stitch_result(res_list, tile_num, voxel_len)
    tile_pos_stitch, tile_refer_id = update_strip_pos_by_MST(tile_pos, tile_shift_arr, tile_shift_loss)
    print(tile_refer_id)

    # calculate new info
    axis_range, voxel_num = calc_axis_range(tile_pos_stitch, dim_elem_num, voxel_len)
    first_last_index = find_first_last_index(tile_pos_stitch, dim_elem_num, axis_range, voxel_len, voxel_num)

    # save data
    save_img_xml_info(info_IO_path, xml_name='meta', img_file_type=img_file_type, ch_num=ch_num,
                      img_data_type=img_data_type, dim_elem_num=dim_elem_num, dim_len=dim_len, voxel_len=voxel_len,
                      tile_pos=tile_pos_stitch, tile_shift_arr=tile_shift_arr, tile_shift_loss=tile_shift_loss,
                      tile_refer_id=tile_refer_id, first_last_index=first_last_index)

    # export data
    export_img_hori_stit(img_path, img_save_path, img_name_format, img_name, ch_num, img_file_type, img_data_type,
                         dim_elem_num, dim_len, voxel_len, tile_pos_stitch, axis_range, first_last_index,
                         img_save_type='tif', whole_img=whole_img, if_blend=if_blend)
