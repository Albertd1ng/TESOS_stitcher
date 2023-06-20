import numpy as np
import cv2


def pyr_down_img(img, times):
    if times == 0:
        return img
    img_down = img.copy()
    while times>0:
        img_down = cv2.pyrDown(img_down)
        img_down = img_down[:, :, ::2]
        times = times - 1
    return img_down


def pyr_down_img_2D(img, times):
    if times == 0:
        return img
    img_down = img.copy()
    while times>0:
        img_down = cv2.pyrDown(img_down)
        img_down = img_down[:, :]
        times = times - 1
    return img_down


def adjust_contrast(img1, img2, max_mean=5):
    img_data_type = img1.dtype
    img1, img2 = img1.astype('float32'), img2.astype('float32')
    m1, m2 = np.mean(img1), np.mean(img2)
    m = np.max((m1, m2, max_mean))
    if img_data_type == 'uint8':
        img1, img2 = np.uint8(np.clip(m / m1 * img1, 0, 255)), np.uint8(np.clip(m / m2 * img2, 0, 255))
    elif img_data_type == 'uint16':
        img1, img2 = np.uint16(np.clip(m / m1 * img1, 0, 65535)), np.uint16(np.clip(m / m2 * img2, 0, 65535))
    return img1, img2
