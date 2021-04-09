# encoding: utf-8
import SimpleITK
import numpy as np
import argparse
import json
import os
import pandas as pd


def traverseFiles(file_dir, suffix='.nii'):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == suffix:
                L.append(os.path.join(root, file))
    return L


def loadNiiFile(file_name, mode='volume'):
    itk_img = SimpleITK.ReadImage(file_name)
    img_array = SimpleITK.GetArrayFromImage(itk_img)  # the array is arranged with [z, y, x]
    dimension = itk_img.GetSize()  # the dimension is arranged with [x, y, z]
    print("Image array: ", dimension)
    # print(img_array.shape)
    _max, _min = np.max(img_array), np.min(img_array)
    print("Volume max : {}, min : {}".format(_max, _min))
    if mode == 'volume':
        regular_array = (img_array - _min) / (_max - _min) * 255
        regular_array = regular_array.astype(np.uint8)
    else:
        # print("Label number : {}". format(len(set(img_array.flatten()))))
        img_array[img_array == 255] = 1
        regular_array = img_array.astype(np.uint8)
    return regular_array


if __name__ == '__main__':

    image_file_path = r'M:\scientific data\lung segment\volume'
    mask_file_path = r'M:\scientific data\lung segment\mask'

    image_files = traverseFiles(image_file_path)
    mask_files = traverseFiles(mask_file_path)

    for index in range(len(image_files)):
        nii_origin_file_name = image_files[index]
        nii_mask_file_name = mask_files[index]
        regular_volume = loadNiiFile(nii_origin_file_name)
        label_volume = loadNiiFile(nii_mask_file_name, mode='label')
        regular_volume.tofile(nii_origin_file_name.split('.')[0] +
                              '_{}_{}_{}_uchar.raw'.format(regular_volume.shape[2],
                                                           regular_volume.shape[1],
                                                           regular_volume.shape[0]))

        label_volume.tofile(nii_origin_file_name.split('.')[0] +
                            '_{}_{}_{}_uchar.label'.format(regular_volume.shape[2],
                                                           regular_volume.shape[1],
                                                           regular_volume.shape[0]))

        # df = pd.DataFrame(columns=['VoxelPos', 'LabelID'])
        #
        # cnt = 0
        # type_number = 2
        # dimension = regular_volume.shape[::-1] # x y z
        # for i in range(type_number):
        #     x = np.array([dimension[0] * dimension[1], dimension[0], 1])
        #     index = np.argwhere(label_volume == i)
        #     index_cls = np.dot(index, x)
        #     type_cls = np.full(index_cls.shape[0], i)
        #     c = np.array([index_cls, type_cls]).transpose()
        #     df_i = pd.DataFrame(c, columns=['VoxelPos', 'LabelID'])
        #     df = df.append(df_i)
        #     # print(df)
        #
        # df.to_csv(nii_origin_file_name.split('.')[0] +
        #           '_{}_{}_{}_uchar.csv'.format(regular_volume.shape[2],
        #                                          regular_volume.shape[1],
        #                                          regular_volume.shape[0]), encoding="gbk", index=False)
        # print('The labeled csv file has been saved.')
