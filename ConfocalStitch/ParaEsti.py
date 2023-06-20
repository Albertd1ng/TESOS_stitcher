import numpy as np


def pyr_down_time_esti(img_shape, v_thred=1000000):
    total = 1
    dim = img_shape.size
    for i in range(dim):
        total = total * img_shape[i]
    n = 0
    while total / ((2**dim)**n) > v_thred:
        n = n + 1
    return n
