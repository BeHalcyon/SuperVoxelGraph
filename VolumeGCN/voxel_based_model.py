

import numpy as np
import argparse
import time
import json
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from rf_model import rfModel
from nn_model import nnModel
from knn_model import knnModel
from svm_model import svmModel

from sklearn.model_selection import train_test_split
from args import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

# def parse_args():
#     parser = argparse.ArgumentParser(description="Do voxel-based model.")
#
#     parser.add_argument('--configure_file', default='../x64/Release/workspace/spheres_supervoxel.json',
#                         help='configure json file')
#
#     args = parser.parse_args()
#     return args

def readVifo(file_name):
    with open(file_name, "r") as f:
        ls = f.readlines()
        for i in range(len(ls)):
            ls[i] = ls[i].strip('\n')

        volume_number = int(ls[0])
        volume_type = ls[1]
        dimension = [int(x) for x in ls[2].split(' ')]
        space = [float(x) for x in ls[3].split(' ')]
        raw_file_name = ls[4]
        return volume_number, volume_type, dimension, space, raw_file_name

def readVolumeRaw(file_name, dtype='uchar'):
    if dtype == 'uchar' or dtype == 'unsigned char' or dtype == 'uint8':
        return np.fromfile(file_name, dtype=np.uint8)
    elif dtype == 'float':
        return np.fromfile(file_name, dtype=np.float32)
    elif dtype == 'ushort' or dtype == 'unsigned short' or dtype == 'uint16':
        return np.fromfile(file_name, dtype=np.uint16)
    elif dtype == 'int' or dtype == 'unsigned int' or dtype == 'uint32':
        return np.fromfile(file_name, dtype=np.int)

def initalVoxelFeatures(hp):
    # 1. read origin raw file
    volume_raw_data = readVolumeRaw(hp.file_path + hp.file_names[0], hp.dtype)
    # 2. read gradient data
    gradient_raw_data = readVolumeRaw(hp.workspace + hp.gradient_file, 'float')

    dimension = hp.dimension
    volume_length = dimension[0] * dimension[1] * dimension[2]

    f_init = np.zeros(shape=(11, volume_length), dtype=np.float32)
    # 4. add scalar feature
    f_init[0, :] = volume_raw_data

    # 5. add gradient feature
    f_init[1, :] = gradient_raw_data

    # 6. add up, down, left, right, front, back features
    volume_raw_data_3d = np.array(volume_raw_data).reshape(dimension[::-1])
    up = np.zeros(dimension[::-1])
    up[0:dimension[2]-1, :, :] = volume_raw_data_3d[1:dimension[2], :, :]
    up[dimension[2]-1, :, :] = volume_raw_data_3d[0, :, :]

    down = np.zeros(dimension[::-1])
    down[0, :, :] = volume_raw_data_3d[dimension[2]-1, :, :]
    down[1:dimension[2], :, :] = volume_raw_data_3d[0:dimension[2]-1, :, :]

    left = np.zeros(dimension[::-1])
    left[:, :, 0] = volume_raw_data_3d[:, :, dimension[0]-1]
    left[:, :, 1:dimension[0]] = volume_raw_data_3d[:, :, 0:dimension[0]-1]

    right = np.zeros(dimension[::-1])
    right[:, :, dimension[0]-1] = volume_raw_data_3d[:, :, 0]
    right[:, :, 0:dimension[0]-1] = volume_raw_data_3d[:, :, 1:dimension[0]]

    front = np.zeros(dimension[::-1])
    front[:, 0, :] = volume_raw_data_3d[:, dimension[1] - 1, :]
    front[:, 1:dimension[1], :] = volume_raw_data_3d[:, 0:dimension[1] - 1, :]

    back = np.zeros(dimension[::-1])
    back[:, dimension[1]-1, :] = volume_raw_data_3d[:, 0, :]
    back[:, 0:dimension[1]-1, :] = volume_raw_data_3d[:, 1:dimension[1], :]

    f_init[2, :] = up.reshape(volume_length)
    f_init[3, :] = down.reshape(volume_length)
    f_init[4, :] = left.reshape(volume_length)
    f_init[5, :] = right.reshape(volume_length)
    f_init[6, :] = front.reshape(volume_length)
    f_init[7, :] = back.reshape(volume_length)

    # 8. add current position
    arg_index = np.argwhere(volume_raw_data_3d != -1000000)
    arg_arr = np.array(arg_index).transpose()
    print(arg_arr.shape)
    f_init[8, :] = arg_arr[0, :]
    f_init[9, :] = arg_arr[1, :]
    f_init[10, :] = arg_arr[2, :]
    f_init = f_init.transpose()

    # 9. normalize feature
    scaler = preprocessing.StandardScaler().fit(f_init)
    f_init = scaler.transform(f_init)

    print(f_init.shape)

    return f_init

def prepareLabeledData(hp, f_init):
    labeled_voxel_volume = None
    # for ground truth data
    if hp.labeled_type == 2:
        train_label_ratio = 0.001
        labeled_voxel_file = hp.workspace + hp.labeled_file  # .label
        labeled_voxel_file = hp.workspace + hp.groundtruth_label_voxel_file  # .raw
        # labeled_voxel_volume = np.load(labeled_voxel_file).flatten()
        labeled_voxel_volume = np.fromfile(labeled_voxel_file, dtype=np.int32)
        print("Start random sample labeled voxels...")
        import random
        sample_index_array = random.sample(range(len(labeled_voxel_volume)),
                                           int(len(labeled_voxel_volume)*train_label_ratio))
        train_data = []
        train_label = labeled_voxel_volume[sample_index_array]
        for i in sample_index_array:
            train_data.append(list(f_init[i]))
        assert len(train_data) == len(train_label)
    else:

        labeled_voxel_file = hp.workspace + hp.labeled_file  # .label
        import SimpleITK
        itk_img = SimpleITK.ReadImage(labeled_voxel_file)
        labeled_voxel_volume = np.array(SimpleITK.GetArrayFromImage(itk_img)).flatten()  # the array is arranged with [z, y, x]

        train_index = np.array(np.where(labeled_voxel_volume>0))[0]
        print(labeled_voxel_volume)
        print(len(train_index))
        np.random.shuffle(train_index)
        train_index = train_index[:int(len(train_index)*0.06)]
        train_data = []
        train_label = []
        for i in train_index:
            train_data.append(list(f_init[i]))
            train_label.append(labeled_voxel_volume[i]-1)
        assert len(train_data) == len(train_label)
    x = np.array(train_data)
    y = np.array(train_label, dtype=np.int32)

    return x, y, labeled_voxel_volume

def main():

    hp = parse_args()

    f_init = initalVoxelFeatures(hp)
    x, y, labeled_voxel_volume = prepareLabeledData(hp, f_init)

    hp.label_type_number = len(set(y))
    hp.labeled_node_number = len(y)
    print('Number of all nodes : ', hp.dimension[0] * hp.dimension[1] * hp.dimension[2])
    print('Number of labeled nodes : ', hp.labeled_node_number)
    print('Number of trained labeled nodes : ', int(hp.labeled_node_number * hp.ratio))
    print('Number of test labeled nodes : ', int(hp.labeled_node_number * (1 - hp.ratio)))

    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=hp.ratio,
                                                                      test_size=1-hp.ratio)  # sklearn.model_selection.

    print(train_data.shape)
    print(train_label.shape)

    type = ''
    print('Input the training model in one of :[\'rf (random forest)\', \'nn (neural network)\', \'svm\', \'knn\']:')
    predictions = None
    volume_raw_data = readVolumeRaw(hp.file_path + hp.file_names[0], hp.dtype)
    types = ['rf', 'nn', 'svm', 'knn']
    logger = logger_config(hp.workspace + "metric_log.txt")
    for type in types:

    # while True:
    #     type = input()
        time_start = time.time()
        if type == 'rf':
            predictions, training_time, test_f1, predicting_time = rfModel(f_init, train_data, train_label, test_data, test_label)
            predictions.tofile(hp.workspace+hp.voxel_based_rf_predict_file)
            labelVoxel2Nii(hp, predictions+1, hp.voxel_based_rf_predict_file.split('.')[0]+'.nii')
            # break
        elif type == 'nn':
            hp.vec_dim = 11
            predictions, training_time, test_f1, predicting_time = nnModel(hp, f_init, train_data, train_label)
            predictions.tofile(hp.workspace+hp.voxel_based_nn_predict_file)
            labelVoxel2Nii(hp, predictions+1, hp.voxel_based_nn_predict_file.split('.')[0] + '.nii')
            # break
        elif type == 'svm':
            predictions, training_time, test_f1, predicting_time = svmModel(f_init, train_data, test_label, test_data, train_label)
            predictions.tofile(hp.workspace+hp.voxel_based_svm_predict_file)
            labelVoxel2Nii(hp, predictions+1, hp.voxel_based_svm_predict_file.split('.')[0] + '.nii')
            # break
        elif type == 'knn':
            predictions, training_time, test_f1, predicting_time = knnModel(f_init, train_data, train_label, test_data, test_label)
            predictions.tofile(hp.workspace+hp.voxel_based_knn_predict_file)
            labelVoxel2Nii(hp, predictions+1, hp.voxel_based_knn_predict_file.split('.')[0] + '.nii')
            # break
        else:
            print('No train model named {}. Please input again.'.format(type))
            continue

        time_end = time.time()
        all_time = int(time_end - time_start)
        hours = int(all_time / 3600)
        minute = int((all_time - 3600 * hours) / 60)
        print()
        print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')

        if hp.labeled_type == 2:
            precision, recall, f1, acc = evaluationForVoxels(labeled_voxel_volume, predictions)
            print("precision score : {}".format(precision))
            print("recall score : {}".format(recall))
            print("f1 score : {}".format(f1))
            print("accuracy score : {}".format(acc))

            # logger = logger_config(hp.workspace + "metric_log.txt")
            logger.info("voxelbased-{}: precision: {}, recall: {}, f1: {}, accuracy : {}, time : {:.2f}, p-time : {:.2f}".
                        format(type, precision, recall, f1, acc, training_time, predicting_time))
        else:
            logger.info("voxelbased-{}: test f1 score: {}, time : {}".
                        format(type, test_f1, training_time))
            continue
            for i in range(hp.label_type_number):
                buf_volume_array = np.zeros(dtype=volume_raw_data.dtype, shape=volume_raw_data.shape)
                buf_index_array = np.array(np.where(predictions == i))
                label_number = len(buf_index_array[0])

                for j in buf_index_array[0]:
                    buf_volume_array[j] = volume_raw_data[j]

                if label_number > 0:
                    buf_volume_array.tofile(
                        hp.workspace + type + '_' + 'feature_part_' + str(i) +'voxel_number_' + str(label_number) + '.raw')
                    print("{}: Cluster {} in {} has {} labels, and it has been saved."
                          .format(type, i, hp.label_type_number, label_number))


if __name__ == '__main__':
    main()