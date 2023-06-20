import numpy as np
from scipy import ndimage as ndi
import cv2
import time


def imshow(img, wait_time=0):
    cv2.imshow('img', cv2.pyrDown(cv2.pyrDown(img)))
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()


def avoid_zero(array, fill=0.01):
    array[array<fill]=fill
    return array


def img_blend(img_array, tile_pos, voxel_num):
    def mask_with_ones(img_):
        return img_ > 0.9
    sigma = np.round(0.05*np.mean(voxel_num))
    img_dtype = str(img_array.dtype)
    img = img_array.astype('float32')
    tile_num = img.shape[2]
    tile_len = img.shape[1::-1]
    whole_img = np.zeros(voxel_num[::-1], dtype='float32')
    whole_img[tile_pos[0, 1]:tile_pos[0, 1] + tile_len[1], tile_pos[0, 0]:tile_pos[0, 0] + tile_len[0]] = img[:, :, 0]
    for i in range(1, tile_num):
        one_img = np.zeros((tile_len[1]+2, tile_len[0]+2), dtype='float32')
        one_img[1:tile_len[1]+1, 1:tile_len[0]+1] = (img[:, :, i]).copy()
        one_img_mask = np.float32(ndi.distance_transform_edt(one_img))
        one_img_mask = one_img_mask[1:tile_len[1]+1, 1:tile_len[0]+1]
        one_img_mask = 1 - (np.e) ** (-(one_img_mask/sigma)**2)
        one_img = one_img[1:tile_len[1]+1, 1:tile_len[0]+1]
        sec_img = (whole_img[tile_pos[i, 1]:tile_pos[i, 1] + tile_len[1], tile_pos[i, 0]:tile_pos[i, 0] + tile_len[0]]).copy()
        sec_img_logic = mask_with_ones(sec_img)
        sec_img_mask = np.float32(sec_img_logic)
        sec_img_mask[sec_img_logic] = sec_img_mask[sec_img_logic] - one_img_mask[sec_img_logic]
        one_img_mask[np.logical_not(sec_img_logic)] = 1
        whole_img[tile_pos[i, 1]:tile_pos[i, 1] + tile_len[1], tile_pos[i, 0]:tile_pos[i, 0] + tile_len[0]] = \
            sec_img_mask * sec_img + one_img_mask * one_img
    whole_img = whole_img - 1
    if img_dtype == 'uint8':
        whole_img = np.uint8(np.clip(whole_img, 0, 255))
    elif img_dtype == 'uint16':
        whole_img = np.uint16(np.clip(whole_img, 0, 65535))
    return whole_img

