

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
from args import parse_args

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
    if dtype == 'uchar':
        return np.fromfile(file_name, dtype=np.uint8)
    elif dtype == 'float':
        return np.fromfile(file_name, dtype=np.float32)
    elif dtype == 'ushort':
        return np.fromfile(file_name, dtype=np.uint16)
    elif dtype == 'int':
        return np.fromfile(file_name, dtype=np.int)
def main():



    hp = parse_args()
    configure_json_file = hp.configure_file

    f = open(configure_json_file)
    json_content = json.load(f)
    file_prefix = json_content["data_path"]["file_prefix"]

    # 0. read vifo file
    vifo_file = json_content["data_path"]["vifo_file"]
    volume_number, volume_type, dimension, space, raw_file_name = readVifo(vifo_file)

    # 1. read origin raw file
    vifo_file_path = vifo_file[:vifo_file.rfind('/') + 1]
    volume_raw_data = readVolumeRaw(vifo_file_path + raw_file_name, volume_type)

    # 2. read gradient data
    gradient_file_name = file_prefix + json_content["file_name"]["gradient_file"]
    gradient_raw_data = readVolumeRaw(gradient_file_name, 'float')

    # 3. read itk labeled csv file
    itk_snap_labeled_voxel_file = file_prefix + json_content["file_name"]["itk_snap_labeled_voxel_file"]
    labeled_voxel_data_frame = pd.read_csv(itk_snap_labeled_voxel_file)

    labeled_voxel_dict = dict()
    for i in range(len(labeled_voxel_data_frame['VoxelPos'])):
        labeled_voxel_dict[labeled_voxel_data_frame['VoxelPos'][i]] = labeled_voxel_data_frame['LabelID'][i]

    volume_length = dimension[0]*dimension[1]*dimension[2]
    f_init = np.zeros(shape=(11, volume_length), dtype=np.float)

    for z in range(dimension[2]):
        for y in range(dimension[1]):
            for x in range(dimension[0]):
                index = z*(dimension[1]*dimension[0]) + y*dimension[0] + x

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
    # print(scaler.mean_)
    # print(scaler.var_)
    f_init = scaler.transform(f_init)
    # print(f_init[:10])
    # print(res[:10])

    print(f_init.shape)

    # 10. get train data
    train_voxel_index_array = labeled_voxel_dict.keys()
    train_data = []
    train_label = []
    for i in train_voxel_index_array:
        train_data.append(list(f_init[i]))
        train_label.append(labeled_voxel_dict[i])
    x = np.array(train_data)
    y = np.array(train_label, dtype=np.int)

    hp.label = len(set(y))
    hp.labeled_node = len(y)
    print('Number of all nodes : ', volume_length)
    print('Number of labeled nodes : ', hp.labeled_node)
    print('Number of trained labeled nodes : ', int(hp.labeled_node * hp.ratio))
    print('Number of test labeled nodes : ', int(hp.labeled_node * (1 - hp.ratio)))

    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=hp.ratio,
                                                                      test_size=1-hp.ratio)  # sklearn.model_selection.

    print(train_data.shape)
    print(train_label.shape)

    type = ''
    print('Input the training model in one of :[\'rf (random forest)\', \'nn (neural network)\', \'svm\', \'knn\']:')
    predictions = None
    while True:
        type = input()
        time_start = time.time()
        if type == 'rf':
            predictions = rfModel(f_init, train_data, train_label, test_data, test_label)
            break
        elif type == 'nn':
            hp.vec_dim = 11
            predictions = nnModel(hp, f_init, train_data, train_label)
            break
        elif type == 'svm':
            predictions = svmModel(f_init, train_data, test_label, test_data, train_label)
            break
        elif type == 'knn':
            predictions = knnModel(f_init, train_data, train_label, test_data, test_label)
            break
        else:
            print('No train model named {}. Please input again.'.format(type))


    print(predictions)
    np.save(hp.predict_labeled_voxel_file, np.array(predictions))

    time_end = time.time()
    all_time = int(time_end - time_start)

    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print()
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')

if __name__ == '__main__':
    main()