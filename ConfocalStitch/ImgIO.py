import cv2
import numpy as np
import os
import nd2
from readlif.reader import LifFile

from AxisRange import calc_max_axis_range_vert_merged
from Blend import img_blend
from merge import MergeSolution


def get_img_from_nd2(whole_img, i, ch_num, ch_th):
    if ch_num == 1:
        old_img = whole_img[i, :, :, :].copy()
    else:
        old_img = whole_img[i, ch_th, :, :, :].copy()
    new_img = np.zeros((old_img.shape[1], old_img.shape[2], old_img.shape[0]), dtype=old_img.dtype)
    for j in range(old_img.shape[0]):
        new_img[:, :, j] = old_img[j, :, :]
    return new_img


def get_img_2D_from_nd2(whole_img, i, ch_num, ch_th, z_th):
    if ch_num == 1:
        img = whole_img[i, z_th, :, :].copy()
    else:
        img = whole_img[i, ch_th, z_th, :, :].copy()
    return img


def get_img_2D_from_nd2_vert(whole_img, ch_num, ch_th, z_th):
    if ch_num == 1:
        img = whole_img[z_th, :, :].copy()
    else:
        img = whole_img[z_th, ch_th, :, :].copy()
    return img


def get_img_from_lif(whole_img, i, ch_th, dim_elem_num):
    if whole_img.info['settings']['BitSize'] == '8':
        img_dtype = 'uint8'
    elif whole_img.info['settings']['BitSize'] == '16':
        img_dtype = 'uint16'
    img = np.zeros((dim_elem_num[1], dim_elem_num[0], dim_elem_num[2]), dtype=img_dtype)
    for j in range(dim_elem_num[2]):
        img[:, :, j] = np.array(whole_img.get_frame(z=j, t=ch_th, m=i), dtype=img_dtype)
    return img


def get_img_2D_from_lif(whole_img, i, ch_th, z_th, dim_elem_num):
    if whole_img.info['settings']['BitSize'] == '8':
        img_dtype = 'uint8'
    elif whole_img.info['settings']['BitSize'] == '16':
        img_dtype = 'uint16'
    img = np.array(whole_img.get_frame(z=z_th, t=ch_th, m=i), dtype=img_dtype)
    return img


def get_img_from_mtif(whole_img, i, dim_elem_num):
    img = np.zeros((dim_elem_num[1], dim_elem_num[0], dim_elem_num[2]), dtype=whole_img.dtype)
    for j in range(dim_elem_num[2]):
        img[:, :, j] = whole_img[i*dim_elem_num[2]+j, :, :]
    return img


def get_img_2D_from_mtif(whole_img, i, z_th, dim_elem_num):
    img = whole_img[i*dim_elem_num[2]+z_th, :, :]
    return img


def import_img_one_tile(img_name_format, img_path, img_name, i, ch_th, dim_elem_num,
                        img_type='tif', img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    voxel_array = np.zeros(tuple(dim_elem_num), dtype=img_data_type)
    for j in range(dim_elem_num[2]):
        one_img_name = os.path.join(img_path, img_name_format % (img_name, i, j, ch_th, img_type))
        # print(one_img_name)
        voxel_array[:, :, j] = cv2.imread(one_img_name, img_mode)
    return voxel_array


def import_img_2D(img_name_format, img_path, img_name, z_th, channel_ordinal,
                  img_type='tif', img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    one_img_name = os.path.join(img_path, img_name_format % (img_name, z_th, channel_ordinal, img_type))
    one_img = cv2.imread(one_img_name, img_mode)
    return one_img


def import_img_2D_tile(img_name_format, img_path, img_name, ordinal, z_th, ch_th,
                       img_type='tif', img_data_type='uint8', img_mode=cv2.IMREAD_GRAYSCALE):
    one_img_name = os.path.join(img_path, img_name_format % (img_name, ordinal, z_th, ch_th, img_type))
    return cv2.imread(one_img_name, img_mode)


def export_img_hori_stit(img_path, img_save_path, img_name_format, img_name, ch_num, img_type, img_data_type,
                         dim_elem_num, dim_len, voxel_len, tile_pos, axis_range, first_last_index,
                         img_save_type='tif', whole_img=None, if_blend=True):
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
            if if_blend:
                this_img_array = np.zeros((dim_elem_num[1], dim_elem_num[0], tile_num), dtype=img_data_type)
                this_tile_pos = np.zeros((tile_num, 2), dtype='int64')
                tile_num_k = 0
                # img_list = []
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
                if img_type == 'nd2':
                    img_2D = get_img_2D_from_nd2(whole_img, k, ch_num, ch_th, z_th)
                elif img_type == 'lif':
                    img_2D = get_img_2D_from_lif(whole_img, k, ch_th, z_th, dim_elem_num)
                elif img_type == 'mtif':
                    img_2D = get_img_2D_from_mtif(whole_img, k, z_th, dim_elem_num)
                elif img_type == 'tif':
                    img_2D = import_img_2D_tile(img_name_format, img_path, img_name, k, z_th, ch_th,
                                                img_type=img_type, img_data_type=img_data_type, img_mode=img_mode)
                if if_blend:
                    this_img_array[:, :, tile_num_k] = img_2D
                    this_tile_pos[tile_num_k, :] = [x_th, y_th]
                    tile_num_k += 1
                    # img_list.append([img_2D, [[x_th, x_th + dim_elem_num[0]], [y_th, y_th + dim_elem_num[1]]]])
                else:
                    this_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0]] = img_2D
            if if_blend:
                this_img = img_blend(this_img_array[:, :, 0:tile_num_k], this_tile_pos[0:tile_num_k, :],
                                     voxel_num[0:2])
                # this_img = MergeSolution(img_list, voxel_num[0:2], np.uint8).do()
            cv2.imwrite(os.path.join(img_save_path, '%s_z%.4d_ch%.2d.%s' % (img_name, img_num, ch_th, img_save_type)),
                        this_img)
            img_num += 1


def export_img_vert_stit(file_path, file_list, img_save_path, axis_range_array, first_last_index, img_name_format,
                         img_name, ch_num, img_file_type, img_data_type):
    if img_data_type == 'uint8':
        img_mode = cv2.IMREAD_GRAYSCALE
    elif img_data_type == 'uint16':
        img_mode = cv2.IMREAD_UNCHANGED
    layer_num = axis_range_array.shape[0]

    xy_axis_min = np.min(axis_range_array[:, :, 0], axis=0)
    axis_range_array[:, :, 0] = axis_range_array[:, :, 0] - xy_axis_min
    axis_range_array[:, :, 1] = axis_range_array[:, :, 1] - xy_axis_min
    xy_axis_max = np.max(axis_range_array[:, :, 1], axis=0)

    img_num = 0
    for i in range(layer_num):
        img_path = os.path.join(file_path, file_list[i])
        if first_last_index[i, 1] - first_last_index[i, 1] <= 0:
            continue
        x_min = axis_range_array[i, 0, 0]
        x_max = axis_range_array[i, 0, 1]
        y_min = axis_range_array[i, 1, 0]
        y_max = axis_range_array[i, 1, 1]
        if img_file_type == 'nd2':
            pass
        elif img_file_type == 'lif':
            pass
        elif img_file_type == 'mtif':
            pass
        elif img_file_type == 'tif':
            for j in range(first_last_index[i, 0], first_last_index[i, 1]):
                for c in range(ch_num):
                    this_img = np.zeros(xy_axis_max[::-1], dtype=img_data_type)
                    img_2D = import_img_2D(img_name_format, img_path, img_name, j, c, img_type=img_file_type,
                                           img_data_type=img_data_type, img_mode=img_mode)
                    this_img[y_min:y_max, x_min:x_max] = img_2D
                    cv2.imwrite(os.path.join(img_save_path, img_name_format % (img_name, img_num, c, img_num)),
                                this_img)
                img_num += 1


def export_img_vert_stit_merged(layer_num, info_IO_path, file_path, file_name_format, img_save_path, img_name_format,
                                img_name, channel_num, img_type, img_data_type, img_num=0):
    if img_data_type == 'uint8':
        img_mode = cv2.IMREAD_GRAYSCALE
    elif img_data_type == 'uint16':
        img_mode = cv2.IMREAD_UNCHANGED
    xy_axis_range, xy_voxel_num = calc_max_axis_range_vert_merged(layer_num,info_IO_path)
    for i in range(layer_num):
        print(i)
        img_path = os.path.join(file_path, file_name_format % (i))
        dim_elem_num = np.load(os.path.join(info_IO_path, 'dim_elem_num_zstitch_%.4d.npy' % (i)))
        axis_range = np.load(os.path.join(info_IO_path, 'axis_range_zstitch_%.4d.npy' % (i)))
        first_last_index = np.load(os.path.join(info_IO_path, 'first_last_index_zstitch_%.4d.npy' % (i)))
        if img_type == 'nd2':
            whole_img = nd2.imread(img_path + '.' + 'nd2')
            for j in range(first_last_index[0], first_last_index[1]):
                for c in range(channel_num):
                    this_img = np.zeros(xy_voxel_num[1::-1], dtype=img_data_type)
                    x_th, y_th = axis_range[0, 0] - xy_axis_range[0, 0], axis_range[1, 0] - xy_axis_range[1, 0]
                    img_2D = get_img_2D_from_nd2_vert(whole_img, channel_num, c, j)
                    this_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0]] = img_2D
                    cv2.imwrite(os.path.join(img_save_path, img_name_format % (img_name, img_num, c, 'tif')),
                                this_img)
                img_num += 1
        elif img_type == 'tif':
            for j in range(first_last_index[0], first_last_index[1]):
                for c in range(channel_num):
                    this_img = np.zeros(xy_voxel_num[1::-1], dtype=img_data_type)
                    x_th, y_th = axis_range[0, 0] - xy_axis_range[0, 0], axis_range[1, 0] - xy_axis_range[1, 0]
                    img_2D = import_img_2D(img_name_format, img_path, img_name, j, c, img_type=img_type,
                                           img_data_type=img_data_type, img_mode=img_mode)
                    this_img[y_th:y_th + dim_elem_num[1], x_th:x_th + dim_elem_num[0]] = img_2D
                    cv2.imwrite(os.path.join(img_save_path, img_name_format % (img_name, img_num, c, img_type)),
                                this_img)
                img_num += 1
