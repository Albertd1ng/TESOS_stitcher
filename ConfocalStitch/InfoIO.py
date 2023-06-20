from xml.etree import ElementTree as ET
from readlif.reader import LifFile
import numpy as np
import re
import os
import math
import nd2
import cv2


def get_img_nd2_info(img):
    dim_elem_num = np.zeros(3, dtype='int64')
    dim_elem_num[0] = img.sizes['X']
    dim_elem_num[1] = img.sizes['Y']
    dim_elem_num[2] = img.sizes['Z']
    voxel_len = np.zeros(3, dtype='float64')
    voxel_len[0] = img.voxel_size().x
    voxel_len[1] = img.voxel_size().y
    voxel_len[2] = img.voxel_size().z
    dim_len = dim_elem_num * voxel_len
    tile_pos_list = img.experiment[0].parameters.points
    tile_num = len(tile_pos_list)
    tile_pos = np.zeros((tile_num, 3), dtype='float64')
    for i in range(tile_num):
        tile_pos[i, :] = np.array(tile_pos_list[i].stagePositionUm, dtype='float64')
    dim_num = img.ndim
    img_data_type = str(img.dtype)
    return dim_elem_num, dim_len, voxel_len, tile_num, tile_pos, dim_num, img_data_type


def get_img_nd2_info_vert(img):
    dim_elem_num = np.zeros(3, dtype='int64')
    dim_elem_num[0] = img.sizes['X']
    dim_elem_num[1] = img.sizes['Y']
    dim_elem_num[2] = img.sizes['Z']
    dim_num = img.ndim
    img_data_type = str(img.dtype)
    return dim_elem_num, dim_num, img_data_type


def get_img_lif_info(img):
    lif_root = img.xml_root
    dim_elem_num = []
    dim_len = []
    for i in lif_root.iter('TimeStampList'):
        i.text = ''
    for i in lif_root.iter('DimensionDescription'):
        dim_elem_num.append(int(i.attrib['NumberOfElements']))
        dim_len.append(np.float64(i.attrib['Length']))
    tile_num = dim_elem_num[3]
    dim_elem_num = np.array(dim_elem_num[:3], dtype='int64')
    dim_len = np.array(dim_len[:3], dtype='float64')
    voxel_len = dim_len / dim_elem_num
    tile_pos = np.zeros((tile_num, 3), dtype='float64')
    this_num = 0
    for i in lif_root.iter('Tile'):
        tile_pos[this_num, 0] = np.float64(i.attrib['PosX'])
        tile_pos[this_num, 1] = np.float64(i.attrib['PosY'])
        this_num = this_num + 1
    return dim_elem_num, dim_len, voxel_len, tile_num, tile_pos


def get_img_txt_info(img_path, txt_path, ch_num):
    def get_xyz_from_str(xyz_str):
        new_xyz_str = re.sub('(\(|\)| )', '', xyz_str)
        # print(new_xyz_str)
        xyz_str_list = new_xyz_str.split(',')
        x, y = int(round(float(xyz_str_list[0]))), int(round(float(xyz_str_list[1])))
        return [x, y]

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

    x_pos_list, y_pos_list = [], []
    with open(txt_path, 'r') as f:
        while one_line := f.readline():
            one_line_list = one_line.split(';')
            if len(one_line_list) != 3:
                continue
            xyz_list = get_xyz_from_str(one_line_list[2])
            x_pos_list.append(xyz_list[0]), y_pos_list.append(xyz_list[1])
    tile_num = len(x_pos_list)
    dim_elem_num = np.zeros(3, dtype='int64')
    file_list = os.listdir(img_path)
    file_list = pop_other_type_file(file_list, 'tif')
    file_num = len(file_list)
    dim_elem_num[2] = int(math.floor(file_num / tile_num / ch_num))
    one_img = cv2.imread(os.path.join(img_path, file_list[0]))
    dim_elem_num[0], dim_elem_num[1] = one_img.shape[1], one_img.shape[0]
    dim_len = dim_elem_num.astype('float64')
    voxel_len = np.array((1, 1, 1), dtype='float64')
    tile_pos = np.zeros([tile_num, 3], dtype='float64')
    tile_pos[:, 0] = x_pos_list
    tile_pos[:, 1] = y_pos_list
    return dim_elem_num, dim_len, voxel_len, tile_num, tile_pos


def get_img_xml_info(xml_path):
    r"""
    XML Form
    ----------
    <Dimensions>
    <DimensionDescription DimID="1" NumberOfElements="512" Length="2.214287e-04"/>
    <DimensionDescription DimID="2" NumberOfElements="512" Length="2.214287e-04"/>
    <DimensionDescription DimID="3" NumberOfElements="126" Length="2.499821e-04"/>
    <DimensionDescription DimID="10" NumberOfElements="1628" Length="1.627000e+03"/>
    </Dimensions>
    <Attachment Name="TileScanInfo" Application="LAS AF" FlipX="0" FlipY="0" SwapXY="0">
    <Tile FieldX="0" FieldY="0" PosX="0.0154545825" PosY="0.0209818193" PosZ="-0.0001240090"/>
    <Tile FieldX="1" FieldY="0" PosX="0.0156538684" PosY="0.0209818193" PosZ="-0.0001240090"/>
    <Tile FieldX="2" FieldY="0" PosX="0.0158531542" PosY="0.0209818193" PosZ="-0.0001240090"/>
    ......
    <Tile FieldX="24" FieldY="0" PosX="0.0146574392" PosY="0.0213803910" PosZ="-0.0001240090"/>
    </Attachment>
    """
    tree = ET.parse(xml_path)
    this_root = tree.getroot()
    dim_elem_num = []
    dim_len = []
    for i in this_root.iter('TimeStampList'):
        i.text = ''
    for i in this_root.iter('DimensionDescription'):
        dim_elem_num.append(int(i.attrib['NumberOfElements']))
        dim_len.append(np.float64(i.attrib['Length']))
    tile_num = dim_elem_num[3]
    dim_elem_num = np.array(dim_elem_num[:3], dtype='int64')
    dim_len = np.array(dim_len[:3], dtype='float64')
    voxel_len = dim_len / dim_elem_num
    tile_pos = np.zeros((tile_num, 3), dtype='float64')
    this_num = 0
    for i in this_root.iter('Tile'):
        tile_pos[this_num, 0] = np.float64(i.attrib['PosX'])
        tile_pos[this_num, 1] = np.float64(i.attrib['PosY'])
        this_num = this_num + 1
    return dim_elem_num, dim_len, voxel_len, tile_num, tile_pos


def save_img_xml_info(info_IO_path, xml_name='meta', img_file_type=None, ch_num=None, img_data_type=None,
                      dim_elem_num=None, dim_len=None, voxel_len=None, tile_pos=None, tile_shift_arr=None,
                      tile_shift_loss=None, tile_refer_id=None, first_last_index=None):
    r"""
    <Data>
        <Image FileType='' Channels='' DataType=''/>
        <Dimensions>
            <Dimension DimID='X' VoxelNum='' VoxelLen='' DimLen=''>
            <Dimension DimID='Y' VoxelNum='' VoxelLen='' DimLen=''>
            <Dimension DimID='Z' VoxelNum='' VoxelLen='' DimLen=''>
        </Dimensions>
        <Tiles TileNum>
            <Tile TileID='0' PosX='' PosY='' PosZ=''/>
            <Tile TileID='1' PosX='' PosY='' PosZ=''/>
            <Tile TileID='2' PosX='' PosY='' PosZ=''/>
            ......
        </Tiles>
        <Stitches>
            <Stitch TileID1='' TileID2='' ShiftX='' ShiftY='' ShiftZ='' Loss='' IfChosen=''>
            <Stitch TileID1='' TileID2='' ShiftX='' ShiftY='' ShiftZ='' Loss='' IfChosen=''>
            <Stitch TileID1='' TileID2='' ShiftX='' ShiftY='' ShiftZ='' Loss='' IfChosen=''>
            ......
        </Stitches>
        <Output FirstZIndex='' LastZIndex=''>
    </Data>
    """
    Data = ET.Element('Data')
    img_attirb = {}
    if img_file_type is not None:
        img_attirb['FileType'] = img_file_type
    if ch_num is not None:
        img_attirb['Channels'] = str(ch_num)
    if img_data_type is not None:
        img_attirb['DataType'] = img_data_type
    ET.SubElement(Data, 'Image', attrib=img_attirb)

    Dimensions = ET.SubElement(Data, 'Dimensions')
    dim_attrib1 = {'DimID': 'X'}
    dim_attrib2 = {'DimID': 'Y'}
    dim_attrib3 = {'DimID': 'Z'}
    if dim_elem_num is not None:
        dim_attrib1['VoxelNum'] = str(dim_elem_num[0])
        dim_attrib2['VoxelNum'] = str(dim_elem_num[1])
        dim_attrib3['VoxelNum'] = str(dim_elem_num[2])
    if voxel_len is not None:
        dim_attrib1['VoxelLen'] = str(voxel_len[0])
        dim_attrib2['VoxelLen'] = str(voxel_len[1])
        dim_attrib3['VoxelLen'] = str(voxel_len[2])
    if dim_len is not None:
        dim_attrib1['DimLen'] = str(dim_len[0])
        dim_attrib2['DimLen'] = str(dim_len[1])
        dim_attrib3['DimLen'] = str(dim_len[2])
    ET.SubElement(Dimensions, 'Dimension', attrib=dim_attrib1)
    ET.SubElement(Dimensions, 'Dimension', attrib=dim_attrib2)
    ET.SubElement(Dimensions, 'Dimension', attrib=dim_attrib3)

    tile_attrib = {}
    if tile_pos is not None:
        tile_num = tile_pos.shape[0]
        tile_attrib['TileNum'] = str(tile_num)
    Tiles = ET.SubElement(Data, 'Tiles', attrib=tile_attrib)
    if tile_pos is not None:
        for i in range(tile_num):
            tile_attrib = {'TileID': str(i),
                           'PosX': str(tile_pos[i, 0]),
                           'PosY': str(tile_pos[i, 1]),
                           'PosZ': str(tile_pos[i, 2])}
            ET.SubElement(Tiles, 'Tile', attrib=tile_attrib)

    if tile_shift_arr is not None:
        Stitches = ET.SubElement(Data, 'Stitches')
        for i in range(tile_num):
            for j in range(tile_num):
                if tile_shift_loss[i, j] > 0:
                    stitch_attrib = {'TileID1': str(i),
                                     'TileID2': str(j),
                                     'ShiftX': str(tile_shift_arr[i, j, 0]),
                                     'ShiftY': str(tile_shift_arr[i, j, 1]),
                                     'ShiftZ': str(tile_shift_arr[i, j, 2]),
                                     'Loss': str(tile_shift_loss[i, j])}
                    if tile_refer_id is not None:
                        if tile_refer_id[j] == i:
                            stitch_attrib['IfChosen'] = 'True'
                        else:
                            stitch_attrib['IfChosen'] = 'False'
                    ET.SubElement(Stitches, 'Stitch', attrib=stitch_attrib)

    if first_last_index is not None:
        output_attrib = {'FirstIndex': str(first_last_index[0]),
                         'LastIndex': str(first_last_index[1])}
        ET.SubElement(Data, 'Output', output_attrib)

    tree = ET.ElementTree(Data)
    tree.write(os.path.join(info_IO_path, xml_name+'.xml'))


def save_img_xml_info_vert(info_IO_path, xml_name='meta', img_file_type=None, ch_num=None, img_data_type=None,
                           dim_elem_num=None, axis_range_array=None, xyz_shift_array=None, first_last_index=None):
    pass