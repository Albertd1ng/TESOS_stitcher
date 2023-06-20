import numpy as np


def find_peak(array, num=10):
    def get_array_unit(array_f, i_, j_, k_):
        x_, y_, z_ = array_f.shape
        # if i_ == 0:
        #     x1 = i_
        # else:
        #     x1 = i_ - 1
        # if i_ == x_ - 1:
        #     x2 = i_ + 1
        # else:
        #     x2 = i_ + 2
        #
        # if j_ == 0:
        #     y1 = j_
        # else:
        #     y1 = j_ - 1
        # if j_ == y_ - 1:
        #     y2 = j_ + 1
        # else:
        #     y2 = j_ + 2
        #
        # if k_ == 0:
        #     z1 = k_
        # else:
        #     z1 = k_ - 1
        # if k_ == z_ - 1:
        #     z2 = k_ + 1
        # else:
        #     z2 = k_ + 2
        if i_ <= 1:
            x1 = 0
        else:
            x1 = i_ - 2
        if i_ >= x_ - 2:
            x2 = x_
        else:
            x2 = i_ + 3

        if j_ <= 1:
            y1 = 0
        else:
            y1 = j_ - 2
        if j_ >= y_ - 2:
            y2 = y_
        else:
            y2 = j_ + 3

        if k_ <= 1:
            z1 = 0
        else:
            z1 = k_ - 2
        if k_ >= z_ - 2:
            z2 = z_
        else:
            z2 = k_ + 3
        return array_f[x1:x2, y1:y2, z1:z2]

    def if_peak(array_f, value):
        x_, y_, z_ = array_f.shape
        for i_ in range(x_):
            for j_ in range(y_):
                for k_ in range(z_):
                    if value < array_f[i_, j_, k_]:
                        return False
        return True

    x, y, z = array.shape
    if_peak_array = np.zeros((x, y, z), dtype='bool')
    for i in range(x):
        for j in range(y):
            for k in range(z):
                array_unit = get_array_unit(array, i, j, k)
                if_peak_array[i, j, k] = if_peak(array_unit, array[i, j, k])
    peak_num = np.sum(if_peak_array)
    # print(peak_num)
    new_array = array * if_peak_array
    max_to_min = np.sort(new_array, axis=None)[::-1]
    this_num = 0
    return_array = -np.ones((num, 3), dtype='int64')
    num = np.min((num, peak_num))
    for i in max_to_min:
        j = np.argwhere(new_array == i)
        for k in j:
            return_array[this_num, 0] = k[1]
            return_array[this_num, 1] = k[0]
            return_array[this_num, 2] = k[2]
            this_num = this_num + 1
            if this_num >= num:
                return return_array
    return return_array


def get_all_possible_shift(shift_array, img_shape, tile_pos, voxel_len, down_multi):
    num = shift_array.shape[0]
    y, x, z = img_shape
    all_shift_array = np.zeros((num*8, 3), dtype='int64')
    all_shift_array[0 * num:1 * num, :] = shift_array - np.array([0, 0, 0])
    all_shift_array[1 * num:2 * num, :] = shift_array - np.array([x, 0, 0])
    all_shift_array[2 * num:3 * num, :] = shift_array - np.array([0, y, 0])
    all_shift_array[3 * num:4 * num, :] = shift_array - np.array([0, 0, z])
    all_shift_array[4 * num:5 * num, :] = shift_array - np.array([0, y, z])
    all_shift_array[5 * num:6 * num, :] = shift_array - np.array([x, 0, z])
    all_shift_array[6 * num:7 * num, :] = shift_array - np.array([x, y, 0])
    all_shift_array[7 * num:8 * num, :] = shift_array - np.array([x, y, z])
    all_shift_array = all_shift_array + np.int64(np.round((tile_pos[0, :] - tile_pos[1, :]) * down_multi / voxel_len))
    return all_shift_array
