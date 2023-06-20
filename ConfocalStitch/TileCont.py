import numpy as np
import os


def judge_tile_cont(dim_len, tile_pos, tile_cont_thre=0.8):
    tile_num = tile_pos.shape[0]
    tile_contact = np.zeros((tile_num, tile_num), dtype='bool')
    if_x = dim_len * np.array([1, 1 - tile_cont_thre, 1 - tile_cont_thre])
    if_y = dim_len * np.array([1 - tile_cont_thre, 1, 1 - tile_cont_thre])
    for i in range(tile_num):
        for j in range(tile_num):
            if i == j:
                tile_contact[i, j] = False
                continue
            if i > j:
                continue
            if np.all(np.abs(tile_pos[i, :] - tile_pos[j, :]) < if_x):
                tile_contact[i, j] = True
                tile_contact[j, i] = True
                continue
            if np.all(np.abs(tile_pos[i, :] - tile_pos[j, :]) < if_y):
                tile_contact[i, j] = True
                tile_contact[j, i] = True
                continue
    return tile_contact
