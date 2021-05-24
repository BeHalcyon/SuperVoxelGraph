# encoding: utf-8
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
import random
from module import Graphs, metricMeasure
import numpy as np


def readVolumeRaw(file_name, dtype='uchar'):
    if dtype == 'uchar' or dtype == 'unsigned char' or dtype == 'uint8':
        return np.fromfile(file_name, dtype=np.uint8)
    elif dtype == 'float':
        return np.fromfile(file_name, dtype=np.float32)
    elif dtype == 'ushort' or dtype == 'unsigned short' or dtype == 'uint16':
        return np.fromfile(file_name, dtype=np.uint16)
    elif dtype == 'int' or dtype == 'unsigned int' or dtype == 'uint32':
        return np.fromfile(file_name, dtype=np.int)


def initSupervoxelFeatures(hp):
    # read graph
    G = Graphs(hp)
    print("The gexf graph data has been loaded.")
    node_num = len(G.nodes())
    hp.node_num = node_num
    labeled_nodes = []
    all_nodes = []
    label_set = set()
    ground_truth_array = []
    if hp.labeled_type != 2:
        for n in range(node_num):
            all_nodes.append(n)
            if G.nodes[str(n)]['cls'] != -1:
                labeled_nodes.append([n, G.nodes[str(n)]['cls']])
                label_set.add(G.nodes[str(n)]['cls'])
        hp.label = len(label_set)
        random.shuffle(labeled_nodes)  # 标签打散

def prepareLabeledData(hp, f_init, ratio = 0.1):

    labeled_voxel_volume = None

    G = Graphs(hp)
    print("The gexf graph data has been loaded.")
    node_num = len(G.nodes())
    hp.node_num = node_num

    # for ground truth data
    if hp.labeled_type == 2:

        train_label_ratio = ratio
        import random
        sample_index_array = random.sample(range(node_num),
                                           int(node_num * train_label_ratio))

        train_data = []
        train_label = []
        for i in sample_index_array:
            train_data.append(list(f_init[i]))
            train_label.append(G.nodes[str(i)]['cls'])

    else:


        train_data = []
        train_label = []
        for i in range(node_num):
            if G.nodes[str(i)]['cls'] >= 0:
                train_data.append(list(f_init[i]))
                train_label.append(G.nodes[str(i)]['cls'])

    x = np.array(train_data)
    y = np.array(train_label, dtype=np.int32)
    return x, y

def main():
    time_start = time.time()
    hp = parse_args()

    # 1. initial supervoxel features
    f_init = np.load(hp.workspace + hp.graph_node_feature_file)
    f_init = f_init[:, :hp.vec_dim]
    # Debug 20210513
    # 1.1 using the neighboring information of super-voxels

    # a. load graph data
    import networkx as nx
    G = nx.read_gexf(hp.dataset)
    # b. load neighbors
    f_init_neighbor = [[]]*len(f_init)
    for a in G.nodes():
        buf = []
        for neighbor in G.neighbors(a):
            buf.append(f_init[int(neighbor)])
        f_init_neighbor[int(a)] = np.average(np.array(buf), axis=0)
    f_init_neighbor = np.array(f_init_neighbor)
    # c. concatenate
    f_init = np.concatenate([f_init, f_init_neighbor], axis=1)
    # print(f_init_neighbor.shape, f_init.shape)
    # exit()




    # 2. split training and test set
    train_ratio = 0.1
    x, y = prepareLabeledData(hp, f_init, train_ratio)

    hp.labeled_node = len(x)
    hp.label_type_number = len(set(y))
    print('Number of all nodes : ', hp.node_num)
    print('Number of labeled nodes : ', hp.labeled_node)
    print('Number of trained labeled nodes : ', int(hp.labeled_node * train_ratio))
    print('Number of test labeled nodes : ', int(hp.labeled_node * (1 - train_ratio)))

    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=hp.ratio,
                                                                      test_size=1-hp.ratio)  # sklearn.model_selection.

    print(train_data.shape)
    print(train_label.shape)

    print('Input the training model in one of :[\'rf (random forest)\', \'nn (neural network)\', \'svm\', \'knn\']:')
    predictions = None

    groundtruth_supervoxel_label_array = np.load(hp.groundtruth_label_supervoxel_file)
    # read label int file
    label_int_data = np.fromfile(hp.workspace+hp.supervoxel_id_file, dtype=np.int32)

    logger = logger_config(hp.workspace + "metric_log.txt")
    types = ['rf', 'nn', 'svm', 'knn']
    for type in types:
    # while True:
    #     type = input()
        saved_voxel_based_predict_file = None
        time_start = time.time()
        if type == 'rf':
            predictions, training_time, _, predicting_time = rfModel(f_init, train_data, train_label, test_data, test_label) # this prediction is supervoxel-based label array
            saved_voxel_based_predict_file = hp.supervoxel_based_rf_predict_file
            # break
        elif type == 'nn':
            # hp.vec_dim = 256
            predictions, training_time, _, predicting_time = nnModel(hp, f_init, train_data, train_label)
            saved_voxel_based_predict_file = hp.supervoxel_based_nn_predict_file

            # voxel_based_predictions = volumeSegmentation(predictions, label_int_data)
            # voxel_based_predictions.tofile(hp.workspace + hp.supervoxel_based_nn_predict_file)
            # labelVoxel2Nii(hp, voxel_based_predictions, hp.voxel_based_nn_predict_file.split('.')[0] + '.nii')
            # break
        elif type == 'svm':
            predictions, training_time, _, predicting_time = svmModel(f_init, train_data, test_label, test_data, train_label)
            saved_voxel_based_predict_file = hp.supervoxel_based_svm_predict_file

            # voxel_based_predictions = volumeSegmentation(predictions, label_int_data)
            # voxel_based_predictions.tofile(hp.workspace + hp.supervoxel_based_svm_predict_file)
            # labelVoxel2Nii(hp, voxel_based_predictions, hp.voxel_based_svm_predict_file.split('.')[0] + '.nii')
            # break
        elif type == 'knn':
            predictions, training_time, _, predicting_time = knnModel(f_init, train_data, train_label, test_data, test_label)
            saved_voxel_based_predict_file = hp.supervoxel_based_knn_predict_file

            # voxel_based_predictions = volumeSegmentation(predictions, label_int_data)
            # voxel_based_predictions.tofile(hp.workspace + hp.supervoxel_based_knn_predict_file)
            # labelVoxel2Nii(hp, voxel_based_predictions, hp.voxel_based_knn_predict_file.split('.')[0] + '.nii')
            # break
        else:
            print('No train model named {}. Please input again.'.format(type))
            continue


        voxel_based_predictions = volumeSegmentation(predictions, label_int_data)
        voxel_based_predictions.tofile(hp.workspace + saved_voxel_based_predict_file)
        labelVoxel2Nii(hp, voxel_based_predictions, saved_voxel_based_predict_file.split('.')[0] + '.nii')

        time_end = time.time()
        all_time = int(time_end - time_start)
        hours = int(all_time / 3600)
        minute = int((all_time - 3600 * hours) / 60)
        print()
        print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')

        if hp.labeled_type == 2:
            precision, recall, f1, acc = evaluationForVoxels(groundtruth_supervoxel_label_array, predictions)
            print("precision score : {}".format(precision))
            print("recall score : {}".format(recall))
            print("f1 score : {}".format(f1))
            print("accuracy score : {}".format(acc))


            logger.info("supervoxelbased-{}: precision: {}, recall: {}, f1: {}, accuracy : {}, time : {:.2f}, p-time : {:.2f}".
                        format(type, precision, recall, f1, acc, training_time, predicting_time))


if __name__ == '__main__':
    main()
