import math
import numpy as np
import cv2
import os
import time

from InfoIO import get_img_txt_info
from TileCont import judge_tile_cont
from ParaEsti import pyr_down_time_esti
from ImgBorder import get_2img_border, get_border_pyr_down
from ImgOvl import get_ovl_img
from LossFunc import loss_func_for_list
from AxisRange import calc_axis_range, find_first_last_index
from HoriStitch import update_strip_pos_by_MST


def get_img_name_list(img_path, img_type='tif'):
    def pop_other_type_file(file_list_, file_type):
        pop_list = []
        file_type = '.' + file_type
        for i, one_file in enumerate(file_list_):
            if file_type not in one_file:
                pop_list.append(i)
                continue
        for i in pop_list[::-1]:
            file_list_.pop(i)
        return file_list_

    file_list = os.listdir(img_path)
    file_list = pop_other_type_file(file_list, img_type)
    file_list.sort()
    return file_list


def import_img_2D_tile(img_path, file_list, i, dim_elem_num, z_th, ch_num, ch_th, img_mode=cv2.IMREAD_GRAYSCALE):
    one_img_name = os.path.join(img_path, file_list[dim_elem_num[2]*(ch_num*i+ch_th)+z_th])
    return cv2.imread(one_img_name, img_mode)


def import_img_one_tile(img_path, file_list, i, ch_num, ch_th, dim_elem_num,
                        img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    for j in range(dim_elem_num[2]):
        one_img_name = os.path.join(img_path, file_list[dim_elem_num[2] * (i * ch_num + ch_th) + j])
        if j == 0:
            img = cv2.pyrDown(cv2.imread(one_img_name, img_mode))
            voxel_array = np.zeros((img.shape[0], img.shape[1], int(math.ceil(dim_elem_num[2]/2))), dtype=img_data_type)
            voxel_array[:, :, int(j/2)] = img
        elif j % 2 == 0:
            img = cv2.pyrDown(cv2.imread(one_img_name, img_mode))
            voxel_array[:, :, int(j/2)] = img
        elif j % 2 == 1:
            continue
    return voxel_array


def pyr_down_big(img, times):
    if times == 0:
        return img
    img_2D = img[:, :, 0]
    for i in range(times):
        img_2D = cv2.pyrDown(img_2D)
    img_down = np.zeros((img_2D.shape[0], img_2D.shape[1], img.shape[2]), dtype=img.dtype)
    for i in range(img_down.shape[2]):
        img_2D = img[:, :, i]
        for j in range(times):
            img_2D = cv2.pyrDown(img_2D)
        img_down[:, :, i] = img_2D
    for i in range(times):
        img_down = img_down[:, :, ::2]
    return img_down


def blur_big(img, kernel_size=3):
    img_blurred = img.copy()
    for i in range(img.shape[2]):
        img_blurred[:, :, i] = cv2.medianBlur(img[:, :, i], 3)
    return img_blurred


def one_stitch(img1, img2, tile_pos, dim_elem_num, dim_len, voxel_len, voxel_range, pyr_down_times):
    img1_list = [img1, ]
    img2_list = [img2, ]
    for i in range(1, pyr_down_times+1):
        img1_list.append(pyr_down_big(img1_list[i-1], 1))
        img2_list.append(pyr_down_big(img2_list[i-1], 1))
    for pdt in range(pyr_down_times, -1, -1):
        if pdt == pyr_down_times:
            x_s, y_s, z_s = 0, 0, 0
            x_pd, y_pd, z_pd = np.int64(np.round(np.array(voxel_range, dtype='int64') / (2 ** pdt)))
        else:
            x_s, y_s, z_s = 2 * np.array((x_s, y_s, z_s), dtype='int64')
            x_pd = y_pd = z_pd = 2 + pdt
        loss_max = -1
        if pdt != 0:
            img1_pd = img1_list[pdt]
            img2_pd = img2_list[pdt]
            down_multi = np.array([img1_pd.shape[1], img1_pd.shape[0], img1_pd.shape[2]], dtype='float32') / np.array(
                [img1.shape[1], img1.shape[0], img1.shape[2]], dtype='float32')
        else:
            img1_pd = img1_list[0]
            img2_pd = img2_list[0]
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
                    ovl1_list, ovl2_list = get_ovl_img(img1_pd, img2_pd, border, False)
                    this_loss = loss_func_for_list(ovl1_list, ovl2_list)
                    if this_loss > loss_max and this_loss > 0.6:
                        loss_max = this_loss
                        x_s, y_s, z_s = x, y, z
        print(pdt+1, 'down times,', 'shift is ', [x_s, y_s, z_s], 'ncc is ', loss_max)
    if loss_max <= 0.6:
        x_s, y_s, z_s = 0, 0, 0
        loss_max = -1
    if np.abs(x_s) > np.abs(voxel_range[0]) or np.abs(y_s) > np.abs(voxel_range[1]) or np.abs(z_s) > np.abs(voxel_range[2]):
        x_s, y_s, z_s = 0, 0, 0
        loss_max = -1
    print(0, 'down times,', 'shift is ', [x_s*2, y_s*2, z_s*2], 'ncc is ', loss_max)
    return [x_s*2, y_s*2, z_s*2], loss_max


def get_stitch_result(res_list, tile_num, voxel_len):
    tile_shift_arr = np.zeros((tile_num, tile_num, 3), dtype='float64')
    tile_shift_loss = -2 * np.ones((tile_num, tile_num), dtype='float64')
    num = 0
    for k in res_list:
        i, j = k[0], k[1]
        res = k[2]
        tile_shift_arr[i, j, :] = np.array(res[0], dtype='float64') * voxel_len
        tile_shift_arr[j, i, :] = -tile_shift_arr[i, j, :]
        tile_shift_loss[i, j] = res[1]
        tile_shift_loss[j, i] = tile_shift_loss[i, j]
        num = num+1
    return tile_shift_arr, tile_shift_loss


def export_img_hori_stit(img_path, img_save_path, img_name_list, ch_num, img_type, img_data_type,
                         dim_elem_num, dim_len, voxel_len, tile_pos, axis_range, first_last_index,
                         img_save_type='tif'):
    if img_data_type=='uint8':
        img_mode=cv2.IMREAD_GRAYSCALE
    elif img_data_type=='uint16':
        img_mode=cv2.IMREAD_UNCHANGED
    tile_num = tile_pos.shape[0]
    voxel_num = np.int64(np.round((axis_range[:, 1] - axis_range[:, 0]) / voxel_len))

    img_num = 0
    for j in range(first_last_index[0], first_last_index[1]):
        for ch_th in range(ch_num):
            this_img = np.zeros((voxel_num[1::-1]), dtype=img_data_type)
            this_z = axis_range[2, 0] + voxel_len[2] * j
            for k in range(tile_num):
                if tile_pos[k, 2] + dim_len[2] < this_z or tile_pos[k, 2] > this_z:
                    continue
                z_th = np.int64(np.round((this_z - tile_pos[k, 2]) / voxel_len[2]))
                if z_th >= dim_elem_num[2] or z_th < 0:
                    continue
                x_th = np.int64(np.round((tile_pos[k, 0] - axis_range[0, 0]) / voxel_len[0]))
                y_th = np.int64(np.round((tile_pos[k, 1] - axis_range[1, 0]) / voxel_len[1]))
                if x_th < 0 or x_th + dim_elem_num[0] > voxel_num[0] or y_th < 0 or y_th + dim_elem_num[1] > \
                        voxel_num[1]:
                    continue
                if img_type == 'tif':
                    img_2D = import_img_2D_tile(img_path, img_name_list, k, dim_elem_num, z_th, ch_num, ch_th,
                                                img_mode=img_mode)
                this_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0]] = img_2D
            cv2.imwrite(os.path.join(img_save_path, 'z%.4d_ch%.2d.%s' % (img_num, ch_th, img_save_type)),
                        this_img)
        img_num += 1


def start_multi_stitch(info_IO_path, info_file_path, img_file_type, img_path, img_save_path,
                       ch_num, ch_th, img_data_type, move_ratio):
    if img_file_type == 'tif':
        dim_elem_num, dim_len, voxel_len, tile_num, tile_pos = get_img_txt_info(img_path, info_file_path, ch_num)
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

    img_name_list = get_img_name_list(img_path, img_type=img_file_type)
    print('number of images: ', len(img_name_list))
    # tile contact
    tile_contact = judge_tile_cont(dim_len, tile_pos)
    print('contact info has been calculated.')

    # info prepare
    voxel_range = np.int64(np.ceil(dim_elem_num * move_ratio))
    print('move ratio:', voxel_range)
    pyr_down_times = pyr_down_time_esti(dim_elem_num, v_thred=100 * 100 * 50)
    print('down sample times:', pyr_down_times)

    time1 = time.time()
    # start multi stitch
    res_list = []
    i_, j_ = -1, -1
    for i in range(tile_num):
        for j in range(tile_num):
            if i >= j:
                continue
            if tile_contact[i, j]:
                if img_file_type == 'tif':
                    if i == i_:
                        img1_ = img1
                    elif i == j_:
                        img1_ = img2
                    else:
                        img1_ = import_img_one_tile(img_path, img_name_list, i, ch_num, ch_th, dim_elem_num,
                                                    img_data_type=img_data_type, img_mode=img_mode)
                    if j == j_:
                        img2_ = img2
                    elif j == i_:
                        img2_ = img1
                    else:
                        img2_ = import_img_one_tile(img_path, img_name_list, j, ch_num, ch_th, dim_elem_num,
                                                    img_data_type=img_data_type, img_mode=img_mode)
                i_, j_ = i, j
                img1 = img1_
                img2 = img2_
                print(img1.shape, img2.shape)
                res_list.append([i, j, one_stitch(img1, img2, tile_pos[[i, j], :]/2,
                                                  np.int64(img1.shape), img1.shape, voxel_len,
                                                  np.int64(voxel_range/2), pyr_down_times-1)])
    time2 = time.time()
    # update pos
    # tile_shift_arr[i, j], j相对于i的位移值
    tile_shift_arr, tile_shift_loss = get_stitch_result(res_list, tile_num, voxel_len)
    tile_pos_stitch, tile_refer_id = update_strip_pos_by_MST(tile_pos, tile_shift_arr, tile_shift_loss)

    # calculate new info
    axis_range, voxel_num = calc_axis_range(tile_pos_stitch, dim_elem_num, voxel_len)
    first_last_index = np.array([0, voxel_num[2]], dtype='int64')
    print(first_last_index)

    # export data
    export_img_hori_stit(img_path, img_save_path, img_name_list, ch_num, img_file_type, img_data_type,
                         dim_elem_num, dim_len, voxel_len, tile_pos_stitch, axis_range, first_last_index,
                         img_save_type='tif')
    time3 = time.time()
    print(time2-time1)
    print(time3-time2)