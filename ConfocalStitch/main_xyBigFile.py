import os
from HoriStitchBigFile import start_multi_stitch

# 必填项
# 文件数据类型，[nd2, lif, tif...]
img_file_type = 'tif'
# 文件路径，如果是tif等图片格式是文件夹，如果是nd2、lif等特殊格式则具体到文件名
img_path = r'H:\20200523_604_Lavision_12x\230521_2_18-19-23'
# 输出文件路径
img_save_path = r'H:\20200523_604_Lavision_12x\230521_2_18-19-23s'
# 通道数
ch_num = 1
# 用第几个通道拼接
ch_th = 0
# 图片bit数，['uint8', 'uint16']
img_data_type = 'uint16'

# 选填项
# 中间文件路径，默认是拼接文件路径的上一级
info_IO_path = os.path.split(img_path)[0]
# 图片计算范围
move_ratio = [0.07, 0.07, 0.02]
# 位置信息文件
info_file_path = r'H:\20200523_604_Lavision_12x\pos.txt'


if __name__ == '__main__':
    start_multi_stitch(info_IO_path, info_file_path, img_file_type, img_path, img_save_path,
                       ch_num, ch_th, img_data_type, move_ratio)
