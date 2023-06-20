import cv2
import numpy as np
from scipy import ndimage as ndi

u8 = np.uint8
u16 = np.uint16
f16 = np.float16
f32 = np.float32

class MergeSolution():
    """
    算法原理：
        传入若干固定尺寸图像以及对应坐标，一个一个往大图像里面放，每次放的时候跟前面已经放好的使用过渡算法过渡。
    """

    def __init__(self, imgs: list, big_img_size, to_imgdtype=u16):
        '''

        :param imgs: list，不止有img，还要有对应坐标，形如这样[[img1,[x1,y1]],[img2,[x2,y2]],....]
        :param big_img_size: 要拼成的大图像的尺寸，比如这样[1000,2000]，分别对应高和宽
        :param to_imgdtype: 默认图像类型，比如uint16
        '''
        self.imgs, self.coord = [], []
        for img, coord in imgs:
            self.imgs.append(img)
            self.coord.append(coord)  # [x_r, y_r]
        self.result = np.zeros([big_img_size[0] + 2, big_img_size[1] + 2], f32)  # 往四周扩一个像素方便处理
        self.mask = np.zeros([big_img_size[0] + 2, big_img_size[1] + 2], bool)
        self.coord = np.array(self.coord) + 1  # 因为扩了一个像素，这里坐标值都加一
        self.to_imgdtype = to_imgdtype

    def do(self):
        # 第一张不用merge，直接放上
        img0 = self.imgs[0].astype(f32)
        xr, yr = self.coord[0]
        self.result[yr[0]:yr[1], xr[0]:xr[1]] = img0
        self.mask[yr[0]:yr[1], xr[0]:xr[1]] = True

        for i in range(1, len(self.imgs)):
            img = self.imgs[i].astype(f32)
            xr, yr = self.coord[i]
            mask = self.mask[yr[0]:yr[1], xr[0]:xr[1]].astype(u8)
            w_bg = ndi.distance_transform_edt(mask)
            mask = self.mask[yr[0] - 1:yr[1] + 1, xr[0] - 1:xr[1] + 1].copy()  # 往外扩一个像素来取mask
            mask = mask == False  # 值反转
            mask[1:-1, 1:-1] = True
            mask = mask.astype(u8)
            w_fg = ndi.distance_transform_edt(mask)
            w_fg = w_fg[1:-1, 1:-1]
            # w_fg = w_fg[1:w_bg.shape[0]+1, 1:w_bg.shape[1]+1]
            wsum = w_bg + w_fg
            w_bg /= wsum
            w_fg /= wsum
            self.result[yr[0]:yr[1], xr[0]:xr[1]] = \
                w_bg * self.result[yr[0]:yr[1], xr[0]:xr[1]] + \
                w_fg * img
            self.mask[yr[0]:yr[1], xr[0]:xr[1]] = True  # 对应mask置True
        self.result = np.round(self.result[1:-1, 1:-1]).astype(self.to_imgdtype)
        return self.result