import numpy as np
import cv2
import os
import random
import nd2
import time
from readlif.reader import LifFile
from multiprocessing import Pool, cpu_count

from FileRename import rename_file_Z_stit, pop_other_type_file, pop_file
from ImgIO import import_img_2D, export_img_vert_stit
from ImgProcess import pyr_down_img_2D, adjust_contrast
from LossFunc import loss_func_z_stitch
from ParaEsti import pyr_down_time_esti
from InfoIO import get_img_nd2_info_vert


def get_stitch_result_vert(res_list):
    index1, index2 = 0, 0
    xy_shift_max = np.zeros(2, dtype='int64')
    loss_max = -1
    for i in res_list:
        j = i[0]
        k = i[1]
        res = i[2].get()
        if res[1] > loss_max:
            loss_max = res[1]
            xy_shift_max = res[0]
            index1 = j
            index2 = k
    return index1, index2, xy_shift_max, loss_max


def calc_xy_shift_by_SIFT(img1, img2, sample_times=200):
    loss_max = -1
    xy_shift_max = np.zeros(2, dtype='int64')

    # sift
    if img1.dtype == 'uint16':
        img1 = np.uint8(np.floor((img1.astype('float32') / 65535 * 255)))
        img2 = np.uint8(np.floor((img2.astype('float32') / 65535 * 255)))
    # cv2.imshow('1', img1)
    # cv2.imshow('2', img2)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    kpts1, des1 = sift.detectAndCompute(img1, None)
    kpts2, des2 = sift.detectAndCompute(img2, None)
    print(len(kpts1), len(kpts2))
    kp1, kp2 = np.float32([kp.pt for kp in kpts1]), np.float32([kp.pt for kp in kpts2])
    if kp1.shape[0] == 0 or kp2.shape[0] == 0:
        return xy_shift_max, loss_max
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good_matches.append((m[0].queryIdx, m[0].trainIdx))
    pts1, pts2 = np.float32([kp1[i, :] for (i, _) in good_matches]), np.float32([kp2[j, :] for (_, j) in good_matches])
    if pts1.shape[0] == 0:
        return xy_shift_max, loss_max

    # RANSAC
    count = 0
    matches_num = pts1.shape[0]
    RANSAC_num = int(np.max((np.min((4, matches_num * 0.1)), 1)))
    while count < sample_times:
        count += 1
        index_list = random.sample(range(matches_num), RANSAC_num)
        xy_shift_all = pts2[index_list, :] - pts1[index_list, :]
        max_shift, min_shift = np.max(xy_shift_all, axis=0), np.min(xy_shift_all, axis=0)
        if any((max_shift - min_shift) > 100):
            continue
        xy_shift = np.int64(np.round(np.mean(xy_shift_all, axis=0)))
        if all(xy_shift == xy_shift_max):
            continue
        ovl1 = img1[np.max((0, -xy_shift[1])):, np.max((0, -xy_shift[0])):]
        ovl2 = img2[np.max((0, xy_shift[1])):, np.max((0, xy_shift[0])):]
        x_range, y_range = np.min((ovl1.shape[1], ovl2.shape[1])), np.min((ovl1.shape[0], ovl2.shape[0]))
        ovl1, ovl2 = ovl1[0:y_range, 0:x_range], ovl2[0:y_range, 0:x_range]
        this_loss = loss_func_z_stitch(ovl1, ovl2)
        if this_loss > loss_max:
            loss_max = this_loss
            xy_shift_max = xy_shift
    return xy_shift_max, loss_max


def calc_xy_shift_by_BF(img1, img2, xy_shift, pyr_down_times):
    for i in range(pyr_down_times, -1, -1):
        loss_max = -1
        if i == 0:
            img1_calc, img2_calc = img1.copy(), img2.copy()
        else:
            img1_calc, img2_calc = pyr_down_img_2D(img1, i), pyr_down_img_2D(img2, i)
        if i == pyr_down_times:
            xy_shift_max = np.zeros(2, dtype='int64')
            range_calc = 10
        else:
            xy_shift = xy_shift_max * 2
            xy_shift_max = np.zeros(2, dtype='int64')
            range_calc = 2 + i
        for x in range(-range_calc, range_calc+1):
            for y in range(-range_calc, range_calc+1):
                this_xy_shift = xy_shift + np.array([x, y], dtype='int64')
                ovl1 = img1_calc[np.max((0, -this_xy_shift[1])):, np.max((0, -this_xy_shift[0])):]
                ovl2 = img2_calc[np.max((0, this_xy_shift[1])):, np.max((0, this_xy_shift[0])):]
                x_range_max = np.min((ovl1.shape[1], ovl2.shape[1]))
                y_range_max = np.min((ovl1.shape[0], ovl2.shape[0]))
                x_range_min, y_range_min = 0, 0
                while x_range_max - x_range_min > 2000:
                    x_range_max = x_range_max - 100
                    x_range_min = x_range_min + 100
                while y_range_max - y_range_min > 2000:
                    y_range_max = y_range_max - 100
                    y_range_min = y_range_min + 100
                ovl1 = ovl1[y_range_min:y_range_max, x_range_min:x_range_max]
                ovl2 = ovl2[y_range_min:y_range_max, x_range_min:x_range_max]
                this_loss = loss_func_z_stitch(ovl1, ovl2)
                if this_loss > loss_max:
                    loss_max = this_loss
                    xy_shift_max = this_xy_shift
        print('%d.th pyr down times' % i, xy_shift_max, loss_max)
    return xy_shift_max, loss_max


def one_stitch_vert(img1, img2):
    pyr_down_times = max(pyr_down_time_esti(np.array(img1.shape), 800*800), pyr_down_time_esti(np.array(img2.shape), 800*800))
    img1 = cv2.medianBlur(img1, 3)
    img2 = cv2.medianBlur(img2, 3)
    img1_down = pyr_down_img_2D(img1, pyr_down_times)
    img2_down = pyr_down_img_2D(img2, pyr_down_times)
    img1_down, img2_down = adjust_contrast(img1_down, img2_down, max_mean=20)
    # cv2.imshow('1', img1_down)
    # cv2.imshow('2', img2_down)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    xy_shift, this_loss = calc_xy_shift_by_SIFT(img1_down.copy(), img2_down.copy())
    print('SIFT: xy_shift is %s, loss is %.8f' % (str(xy_shift), this_loss))
    xy_shift, this_loss = calc_xy_shift_by_BF(img1, img2, xy_shift, pyr_down_times)
    return xy_shift, this_loss


def start_vertical_stit_merged(file_path, img_save_path, img_name_format, info_IO_path, img_name, ch_num, ch_th,
                               img_file_type, img_data_type, overlap_ratio, if_rename_file, pro_num):
    if img_data_type == 'uint8':
        img_mode = cv2.IMREAD_GRAYSCALE
    elif img_data_type == 'uint16':
        img_mode = cv2.IMREAD_UNCHANGED
    file_list = os.listdir(file_path)
    if img_file_type == 'nd2':
        file_list = pop_other_type_file(file_path, file_list, img_file_type)
    elif img_file_type == 'lif':
        file_list = pop_other_type_file(file_path, file_list, img_file_type)
    elif img_file_type == 'mtif':
        file_list = pop_file(file_path, file_list)
    elif img_file_type == 'tif':
        file_list = pop_file(file_path, file_list)
    else:
        print('input error')
        return
    layer_num = len(file_list)
    if pro_num == -1:
        pro_num = int(np.ceil(0.5 * cpu_count()))

    xy_shift_array = np.zeros((layer_num, 2), dtype='float64')
    axis_range_array = np.zeros((layer_num, 2, 2), dtype='int64')
    first_last_index = np.zeros((layer_num, 2), dtype='int64')
    dim_elem_num = np.zeros((layer_num, 3), dtype='int64')
    for i in range(layer_num-1):
        time1 = time.perf_counter()
        img_path1 = os.path.join(file_path, file_list[i])
        img_path2 = os.path.join(file_path, file_list[i+1])
        if img_file_type == 'nd2':
            pass
        elif img_file_type == 'lif':
            pass
        elif img_file_type == 'mtif':
            pass
        elif img_file_type == 'tif':
            file_list1 = pop_other_type_file(os.listdir(img_path1), img_file_type)
            file_list2 = pop_other_type_file(os.listdir(img_path2), img_file_type)
            dim_elem_num[i, 2] = np.int64(np.floor(len(file_list1) / ch_num))
            dim_elem_num[i+1, 2] = np.int64(np.floor(len(file_list2) / ch_num))
            if if_rename_file:
                if i == 0:
                    rename_file_Z_stit(img_name_format, img_path1, img_name, dim_elem_num[i, 2], ch_num,
                                       img_type=img_file_type)
                rename_file_Z_stit(img_name_format, img_path2, img_name, dim_elem_num[i+1, 2], ch_num,
                                   img_type=img_file_type)
            if i == 0:
                img1 = import_img_2D(img_name_format, img_path1, img_name, dim_elem_num[i, 2] - 1, ch_th,
                                     img_type=img_file_type, img_data_type=img_data_type, img_mode=img_mode)
                dim_elem_num[i, 0], dim_elem_num[i, 1] = img1.shape[1], img1.shape[0]
                first_last_index[i, 1] = dim_elem_num[i, 2]
                axis_range_array[i, 0, 1] = dim_elem_num[i, 0]
                axis_range_array[i, 1, 1] = dim_elem_num[i, 1]
            img2 = import_img_2D(img_name_format, img_path2, img_name, dim_elem_num[i + 1, 2] - 1, ch_th,
                                 img_type=img_file_type, img_data_type=img_data_type, img_mode=img_mode)
            dim_elem_num[i+1, 0], dim_elem_num[i+1, 1] = img2.shape[1], img2.shape[0]
            first_last_index[i+1, 1] = dim_elem_num[i+1, 2]

        ovl_num = int(np.ceil(np.min((dim_elem_num[i, 2], dim_elem_num[i+1, 2])) * overlap_ratio))
        step1 = 7 # max(int(ovl_num/5), 5)
        step2 = 5 # max(step1-2, 3)
        print(step1, step2)
        pool1 = Pool(processes=pro_num)
        res_list = []
        if img_file_type == 'nd2':
            pass
        elif img_file_type == 'lif':
            pass
        elif img_file_type == 'mtif':
            pass
        elif img_file_type == 'tif':
            for j in range(dim_elem_num[i, 2] - 1, dim_elem_num[i, 2] - ovl_num - 1, -step1):
                img1 = import_img_2D(img_name_format, img_path1, img_name, j, ch_th,
                                     img_type=img_file_type, img_data_type=img_data_type, img_mode=img_mode)
                for k in range(0, ovl_num, step2):
                    img2 = import_img_2D(img_name_format, img_path2, img_name, k, ch_th,
                                         img_type=img_file_type, img_data_type=img_data_type, img_mode=img_mode)
                    res_list.append([j, k, pool1.apply_async(one_stitch_vert, args=(img1, img2))])
        pool1.close()
        pool1.join()
        index1, index2, xy_shift_max, loss_max = get_stitch_result_vert(res_list)
        print('#####################################################')
        print('111111', index1, index2, xy_shift_max, loss_max)

        pool2 = Pool(processes=pro_num)
        res_list = []
        if img_file_type == 'nd2':
            pass
        elif img_file_type == 'lif':
            pass
        elif img_file_type == 'mtif':
            pass
        elif img_file_type == 'tif':
            for j in range(index1, index1+1):
                img1 = import_img_2D(img_name_format, img_path1, img_name, j, ch_th,
                                     img_type=img_file_type, img_data_type=img_data_type, img_mode=img_mode)
                for k in range(max(index2-step2+1, 0), min(index2+step2, dim_elem_num[i+1, 2]-1)):
                    img2 = import_img_2D(img_name_format, img_path2, img_name, k, ch_th,
                                         img_type=img_file_type, img_data_type=img_data_type, img_mode=img_mode)
                    res_list.append([j, k, pool2.apply_async(one_stitch_vert, args=(img1, img2))])
        pool2.close()
        pool2.join()
        index1, index2, xy_shift_max, loss_max = get_stitch_result_vert(res_list)
        print('#####################################################')
        print('222222', index1, index2, xy_shift_max, loss_max)

        first_last_index[i, 1] = index1
        first_last_index[i+1, 0] = index2
        xy_shift_array[i+1, :] = xy_shift_max
        axis_range_array[i+1, :, 0] = axis_range_array[i, :, 0] + xy_shift_max
        axis_range_array[i+1, :, 1] = axis_range_array[i+1, :, 0] + dim_elem_num[i+1, :2]
        time2 = time.perf_counter()
        print('time', time2-time1)
    export_img_vert_stit(file_path, file_list, img_save_path, axis_range_array, first_last_index, img_name_format,
                         img_name, ch_num, img_file_type, img_data_type)
