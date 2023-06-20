import numpy as np


def loss_func_z_stitch(ovl1, ovl2):
    ovl1, ovl2 = ovl1.astype('float32'), ovl2.astype('float32')
    ovl1, ovl2 = ovl1 - ovl1.mean(), ovl2 - ovl2.mean()
    loss = (ovl1 * ovl2).mean()/(ovl1.std() * ovl2.std())
    return loss


def loss_func_for_list(ovl1_list, ovl2_list):
    ovl_num = len(ovl1_list)
    if ovl_num == 0:
        return -1
    weight = np.zeros(ovl_num, dtype='float32')
    for i in range(ovl_num):
        weight[i] = ovl1_list[i].size
    weight = weight / weight.sum()
    loss = 0
    for i in range(ovl_num):
        ovl1, ovl2 = ovl1_list[i].astype('float32'), ovl2_list[i].astype('float32')
        ovl1, ovl2 = ovl1 - ovl1.mean(), ovl2 - ovl2.mean()
        a = (ovl1 * ovl2).mean()
        b = ovl1.std() * ovl2.std()
        if b == 0:
            return -1
        loss = loss + weight[i] * a / b
    if loss <= 0.2:
        return -1
    return loss
