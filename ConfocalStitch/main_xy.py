import os
from HoriStitch import start_multi_stitch

# 必填项
# 文件数据类型，[nd2, lif, tif...]
img_file_type = 'tif'
# 文件路径，如果是tif等图片格式是文件夹，如果是nd2、lif等特殊格式则具体到文件名
img_path = r'D:\Albert\Data\ZhaoLab\Image\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G'
# 输出文件路径
img_save_path = r'D:\Albert\Data\ZhaoLab\Image\Thy1_EGFP_stitched_unblended'
# 通道数
ch_num = 2
# 用第几个通道拼接
ch_th = 0
# 图片bit数，['uint8', 'uint16']
img_data_type = 'uint8'
# 是否有位置信息
if_pos_info = True

# 选填项
# 中间文件路径，默认是拼接文件路径的上一级
info_IO_path = os.path.split(img_path)[0]
# 图片计算范围
move_ratio = [0.05, 0.05, 0]
# 是否为稀疏数据
if_sparce = False
# 是否为高噪点数据
if_high_noise = False
# 最大进程数，根据内存大小和文件大小决定，-1则为可调用内存数的一半
pro_num = 3
# 是否启用图像融合
if_blend = False

# 如果是tif图片格式数据则选填下面选项
if img_file_type == 'tif':
    # 文件命名格式
    img_name_format = r'%s_t%.4d_z%.4d_ch%.2d.%s'  # %(img_name, tile_th, z, ch_th, img_type)
    # xml位置信息文件
    info_file_path = r'D:\Albert\Data\ZhaoLab\Image\20220219_Thy1_EGFP_M_high_resolution_40X_10overlap_50G\MetaData\Region 1.xml'
    # 图片名
    img_name = 'Region'
    # 是否重命名文件
    if_rename_file = True

# 如果是mtif图片格式数据则选填下面选项
elif img_file_type == 'mtif':
    # 文件命名格式
    img_name_format = r'%s_ch%.2d.%s'  # %(img_name, ch_th, img_type)
    # xml位置信息文件
    info_file_path = r'E:\DJY\DATA\multipage_tif\MetaData\Region 1.xml'
    # 图片名
    img_name = 'Region'
    # 是否重命名文件
    if_rename_file = True

else:
    img_name_format = r''
    info_file_path = r''
    img_name = ''
    if_rename_file = False


if __name__ == '__main__':
    start_multi_stitch(info_IO_path, info_file_path, img_file_type, img_path, img_save_path, img_name_format,
                       img_name, ch_num, ch_th, img_data_type, if_pos_info, move_ratio, if_sparce, if_high_noise,
                       if_rename_file, if_blend, pro_num)



