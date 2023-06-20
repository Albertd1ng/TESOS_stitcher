import numpy as np


def calc_axis_range(tile_pos, dim_elem_num, voxel_len):
    axis_range = np.zeros((3, 2))
    axis_range[0, 0], axis_range[0, 1] = np.min(tile_pos[:, 0]), np.max(tile_pos[:, 0]) + voxel_len[0] * dim_elem_num[0]
    axis_range[1, 0], axis_range[1, 1] = np.min(tile_pos[:, 1]), np.max(tile_pos[:, 1]) + voxel_len[1] * dim_elem_num[1]
    axis_range[2, 0], axis_range[2, 1] = np.min(tile_pos[:, 2]), np.max(tile_pos[:, 2]) + voxel_len[2] * dim_elem_num[2]
    voxel_num = np.int64(np.round((axis_range[:, 1] - axis_range[:, 0]) / voxel_len))
    return axis_range, voxel_num


def find_first_last_index(tile_pos, dim_elem_num, axis_range, voxel_len, voxel_num):
    first_last_index = np.array([voxel_num[2], 0], dtype='int64')
    tile_num = tile_pos.shape[0]
    for i in range(voxel_num[2]):
        num_one_layer = 0
        this_z = axis_range[2, 0] + voxel_len[2] * i
        for j in range(tile_num):
            if tile_pos[j, 2] + voxel_len[2] * dim_elem_num[2] < this_z:
                continue
            if tile_pos[j, 2] > this_z:
                continue
            z_th = np.int32(np.round((this_z - tile_pos[j, 2]) / voxel_len[2]))
            if z_th >= dim_elem_num[2]:
                continue
            num_one_layer += 1
        if num_one_layer == tile_num:
            if i <= first_last_index[0]:
                first_last_index[0] = i
            if i+1 >= first_last_index[1]:
                first_last_index[1] = i + 1
    return first_last_index


def calc_max_axis_range_vert_merged(layer_num,save_path):
    axis_range_array = np.zeros((layer_num, 4), dtype='int64')
    xy_axis_range=np.zeros((2, 2), dtype='int64')
    for i in range(layer_num):
        axis_range_array[i, :] = np.load(save_path+r'\axis_range_zstitch_%.4d.npy' % (i)).reshape((1, -1))
    xy_axis_range[0, 0], xy_axis_range[0, 1] = np.min(axis_range_array[:, 0]), np.max(axis_range_array[:, 1])
    xy_axis_range[1, 0], xy_axis_range[1, 1] = np.min(axis_range_array[:, 2]), np.max(axis_range_array[:, 3])
    xy_voxel_num = np.int64(xy_axis_range[:, 1] - xy_axis_range[:, 0])
    return xy_axis_range, xy_voxel_num
