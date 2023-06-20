from VertStitchMerged import start_vertical_stit_merged

# 图片类型
img_file_type = 'tif'  # {'tif', 'nd2', 'lif', 'mtif'}
# 文件路径，如果是tif或mtif格式文件放在文件夹中，如果是tif或nd2则为文件，按照命名顺序排列
file_path = r'E:\WYL\371_ZYH\S'
# 拼接后的图片保存路径
img_save_path = r'E:\WYL\371_ZYH\R'
# 通道数
ch_num = 1
# 基于第几个通道进行拼接，假如有3个通道，编号为0，1，2
ch_th = 0

# 可选参数，一般情况下默认即可
# 图片数据类型
img_data_type = 'uint8'  # {'uint8', 'uint16'}
# 计算时考虑的重叠比率
overlap_ratio = 0.8  # z
# 中间文件储存路径，默认文件路径
info_IO_path = file_path
# 最大进程数
pro_num = 1

# 文件格式为tif时选择
if img_file_type == 'tif':
    # 图片命名格式
    img_name_format = '%s_z%.4d_ch%.2d.%s'  # %(img_name,z,ch_th,img_type)
    # 图片名
    img_name = 'img'
    # 是否重命名文件，如果选否，认真填写图片命名格式img_name_format和图片名
    if_rename_file = True

elif img_file_type == 'mtif':
    # 图片命名格式
    img_name_format = '%s_ch%.2d.%s'  # %(img_name,ch_th,img_type)
    # 图片名
    img_name = 'Region'
    # 是否重命名文件，如果选否，认真填写图片命名格式img_name_format和图片名
    if_rename_file = True

else:
    img_name_format = ''
    img_name = ''
    if_rename_file = False

if __name__ == '__main__':
    start_vertical_stit_merged(file_path, img_save_path, img_name_format, info_IO_path, img_name, ch_num, ch_th,
                               img_file_type, img_data_type, overlap_ratio, if_rename_file, pro_num)
