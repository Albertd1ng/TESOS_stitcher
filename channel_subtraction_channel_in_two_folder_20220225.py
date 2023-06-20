import cv2
import numpy as np
import os
from multiprocessing import Pool
import time

############参数############
green_channel_folder = r'C:\Users\likel\Desktop\1'
red_channel_folder = r'C:\Users\likel\Desktop\2'
result_folder = r'C:\Users\likel\Desktop\test'
###########################


def sub(g, r):
    """
    两张图片按照一定规则相减：
    if g>= r:
        g-r
    else:
        0
    :param g: 绿色通道
    :param r: 红色通道
    :return:
    """
    g = g.astype(np.float32)
    r = r.astype(np.float32)
    res = g - r
    res = np.clip(res, 0, None)
    return res


def run(k, green_path, red_path):
    res_path = os.path.join(result_folder, '{:0>3d}.tif'.format(k))
    green = cv2.imread(green_path, cv2.IMREAD_UNCHANGED)
    red = cv2.imread(red_path, cv2.IMREAD_UNCHANGED)
    res = sub(green, red)
    cv2.imwrite(res_path, res, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    print(f'正在处理第{k}图片')


if __name__ == "__main__":
    start_time = time.time()
    green_pics = os.listdir(green_channel_folder)
    red_pics = os.listdir(red_channel_folder)
    green_pics_path = [os.path.join(green_channel_folder, name) for name in green_pics]  # 列表生成式，生成绿色通道文件列表
    red_pics_path = [os.path.join(red_channel_folder, name) for name in red_pics]  # 列表生成式，生成红色通道文件列表
    green_pics_path.sort()  # 排序
    red_pics_path.sort()  # 排序
    p = Pool(60)
    for k, (green, red) in enumerate(zip(green_pics_path, red_pics_path)):
        p.apply_async(run, args=(k, green, red))
    p.close()
    p.join()
    end_time = time.time()
    print('处理完成，共耗时{}秒'.format(end_time - start_time))

