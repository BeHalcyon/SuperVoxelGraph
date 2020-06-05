# -*- coding: utf-8 -*-

# After generating the npy array from semi-supervise GCN,
# this py file depart the volume to the corrsponding parts.


import argparse
import json
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Translate the predict result to segmented result.")

    parser.add_argument('--configure_file', default='../x64/Release/workspace/spheres_supervoxel.json',
                        help='configure json file')

    args = parser.parse_args()
    return args


import numpy as np


# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
#
# for i in range(len(labels)):
#     print(i, labels[i])
#
# dimension = [480, 720, 120] # x y z
# file_path = 'I:\\science data\\4 Combustion\\jet_0051\\'
# histogram_size = 256
#
# volume_array = np.fromfile(file_path + 'jet_mixfrac_0051.raw', dtype=np.float32)
# label_array = np.fromfile(file_path + 'jet_mixfrac_0051_label.raw', dtype=np.int)
# for i in range(-1, n_clusters_):
#
#     buf_volume_array = np.zeros(dtype=np.float32, shape=volume_array.shape)
#     buf_index_array = np.array(np.where(labels == i))
#     # print(buf_index_array)
#     for j in buf_index_array[0]:
#         buf = np.where(label_array == j)
#         # print(buf)
#         buf_volume_array[buf] = volume_array[buf]
#
#     buf_volume_array.tofile('jet_mixfrac_0051_super_voxles'+str(len(buf_index_array[0]))+'_part_'+str(i)+'.raw')
#     print("Cluster ", i , " in ", n_clusters_, " has been saved.")

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
        return np.fromfile(file_name, dtype=np.float)
    elif dtype == 'ushort':
        return np.fromfile(file_name, dtype=np.uint16)
    elif dtype == 'int':
        return np.fromfile(file_name, dtype=np.int)


if __name__ == "__main__":
    start = time.clock()
    hp = parse_args()
    f = open(hp.configure_file)
    json_content = json.load(f)

    file_prefix = json_content["data_path"]["file_prefix"]
    label_raw_file = file_prefix + json_content["file_name"]["label_file"]
    labeled_gexf_file = file_prefix + json_content["file_name"]["labeled_graph_file"]

    # read vifo file
    vifo_file = json_content["data_path"]["vifo_file"]
    volume_number, volume_type, dimension, space, raw_file_name = readVifo(vifo_file)

    # read origin raw file
    vifo_file_path = vifo_file[:vifo_file.rfind('/') + 1]
    volume_raw_data = readVolumeRaw(vifo_file_path + raw_file_name, volume_type)

    # read label int file
    label_int_data = readVolumeRaw(label_raw_file, 'int')

    # read semi-supervise file TODO: integrate the labeled file to json file.
    semisupervise_result_file = '../VolumeGCN/labeled.npy'
    labels = np.load(semisupervise_result_file)


    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    print("Number of labeled type : {}".format(n_clusters_))

    for i in range(-1, n_clusters_):
        buf_volume_array = np.zeros(dtype=volume_raw_data.dtype, shape=volume_raw_data.shape)
        buf_index_array = np.array(np.where(labels == i))
        label_number = len(buf_index_array[0])

        for j in buf_index_array[0]:
            buf = np.where(label_int_data == j)
            # print(buf)
            buf_volume_array[buf] = volume_raw_data[buf]
        if len(buf_index_array[0]) > 0:
            buf_volume_array.tofile(
                file_prefix + str(label_number) + '_part_' + str(i) + '.raw')
            print("Cluster {} in {} has {} labels, and it has been saved.".format(i, n_clusters_, label_number))