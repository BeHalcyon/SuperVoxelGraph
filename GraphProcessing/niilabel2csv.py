# -*- coding: utf-8 -*-

import SimpleITK
import numpy as np
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Translate the nii or nii.gz file to labeled csv file.")

    parser.add_argument('--configure_file', default='../x64/Release/workspace/tooth_supervoxel.json',
                        help='configure json file')

    args = parser.parse_args()
    return args


hp = parse_args()

f = open(hp.configure_file)
json_content = json.load(f)

itk_snap_labeled_nii_file = json_content["data_path"]["vifo_file"]
itk_snap_labeled_nii_file = itk_snap_labeled_nii_file[:-4] + "nii.gz"
print(itk_snap_labeled_nii_file)


file_prefix = json_content["data_path"]["file_prefix"]
itk_snap_labeled_voxel_file = file_prefix + json_content["file_name"]["itk_snap_labeled_voxel_file"]

itk_img = SimpleITK.ReadImage(itk_snap_labeled_nii_file)
img_array = SimpleITK.GetArrayFromImage(itk_img)
dimension = itk_img.GetSize()
print("Image array: ", dimension)  # 读取图像大小
type_number = np.max(img_array)
print("Label type number : {}".format(type_number))

import pandas as pd

df = pd.DataFrame(columns=['VoxelPos', 'LabelID'])

cnt = 0
for i in range(type_number):
    x = np.array([dimension[0] * dimension[1], dimension[0], 1])
    index = np.argwhere(img_array == i + 1)
    index_cls = np.dot(index, x)
    type_cls = np.full(index_cls.shape[0], i)
    c = np.array([index_cls, type_cls]).transpose()
    # print(c)
    df_i = pd.DataFrame(c, columns=['VoxelPos', 'LabelID'])
    df = df.append(df_i)
    # print(df)

df.to_csv(itk_snap_labeled_voxel_file, encoding="gbk", index=False)
print('The labeled csv file has been saved.')
