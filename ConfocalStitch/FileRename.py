import os
import math


def rename_file(img_name_format, img_path, img_name, z_num, channel_num, img_type='tif'):
    file_list = os.listdir(img_path)
    file_list = pop_other_type_file(file_list, img_type)
    file_num = len(file_list)
    tile_num = int(math.floor(file_num / z_num / channel_num))
    this_file_num = 0
    old_name_list = []
    new_name_list = []
    for i in range(tile_num):
        for z in range(z_num):
            for c in range(channel_num):
                old_name = os.path.join(img_path, file_list[this_file_num])
                new_name = os.path.join(img_path, img_name_format % (img_name, i, z, c, img_type))
                old_name_list.append(old_name)
                new_name_list.append(new_name)
                this_file_num += 1
    try:
        for i in range(len(old_name_list)):
            os.rename(old_name_list[i], new_name_list[i])
    except Exception as e:
        print(e)
        for i in range(len(old_name_list)):
            if os.path.exists(new_name_list[i]):
                os.rename(new_name_list[i], old_name_list[i])


def rename_file_mtif(img_name_format, img_path, img_name, ch_num):
    file_list = os.listdir(img_path)
    file_list = pop_other_type_file(file_list, 'tif')
    file_num = len(file_list)
    this_file_num = 0
    old_name_list = []
    new_name_list = []
    for i in range(ch_num):
        old_name = os.path.join(img_path, file_list[this_file_num])
        new_name = os.path.join(img_path, img_name_format % (img_name, i, 'tif'))
        old_name_list.append(old_name)
        new_name_list.append(new_name)
        this_file_num += 1
    try:
        for i in range(len(old_name_list)):
            os.rename(old_name_list[i], new_name_list[i])
    except Exception as e:
        print(e)
        for i in range(len(old_name_list)):
            if os.path.exists(new_name_list[i]):
                os.rename(new_name_list[i], old_name_list[i])


def rename_file_Z_stit(img_name_format, img_path, img_name, z_num, channel_num, img_type='tif'):
    file_list = os.listdir(img_path)
    file_list = pop_other_type_file(file_list, img_type)
    file_num = len(file_list)
    this_file_num = 0
    old_name_list = []
    new_name_list = []
    for z in range(z_num):
        for c in range(channel_num):
            old_name = os.path.join(img_path, file_list[this_file_num])
            new_name = os.path.join(img_path, img_name_format % (img_name, z, c, img_type))
            old_name_list.append(old_name)
            new_name_list.append(new_name)
            this_file_num += 1
    try:
        for i in range(len(old_name_list)):
            os.rename(old_name_list[i], new_name_list[i])
    except Exception as e:
        print(e)
        for i in range(len(old_name_list)):
            if os.path.exists(new_name_list[i]):
                os.rename(new_name_list[i], old_name_list[i])


def pop_other_type_file(file_list, file_type):
    pop_list = []
    file_type = '.' + file_type
    for i, one_file in enumerate(file_list):
        if file_type not in one_file:
            pop_list.append(i)
            continue
    for i in pop_list[::-1]:
        file_list.pop(i)
    return file_list


def pop_file(img_path, file_list):
    pop_list = []
    for i, one_file in enumerate(file_list):
        if os.path.isfile(os.path.join(img_path, one_file)):
            pop_list.append(i)
    for i in pop_list[::-1]:
        file_list.pop(i)
    return file_list
