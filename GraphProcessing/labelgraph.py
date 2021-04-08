# -*- coding: utf-8 -*-
# 给图数据加标签
import networkx as nx
import argparse
import json
import sys
import os
import time
import csv
import numpy as np

from tools import parse_args

def saveGraph(G, graph_file_name):
    nx.write_gexf(G, graph_file_name)
    print("Labeled SuperGraph has been saved.")


# 读取labeled文件
def loadLabelId(csv_file_name):
    labeled_data = []
    labeled_id = []
    with open(csv_file_name) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        labeled_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            labeled_data.append(row[0])
            labeled_id.append(int(row[1]))

    labeled_data = [int(x) for x in labeled_data]  # 将数据从string形式转换为float形式
    labeled_data = np.array(labeled_data)  # 将list数组转化成array数组便于查看数据结构

    label_type_number = len(set(labeled_id))
    labeled_id = np.array(labeled_id)
    return labeled_data, labeled_id, label_type_number


# label graph for multiple csv file
def addLabelForGraph(G, label_csv_file, label_id : int):
    labeled_data, _ = loadLabelId(label_csv_file)
    # test_label_data = loadLabelId("labeled_test_outter_super_voxels.csv")
    for x in labeled_data:
        G.add_node(x, cls=label_id)


# label graph for only one csv file
def addLabelForGraph(G, label_csv_file, labeled_volume_file_name: str):
    # load csv file and return the volume index and its type
    labeled_data, labeled_id, label_type_number = loadLabelId(label_csv_file)
    # load labeled int raw file
    labeled_volume_data = np.fromfile(labeled_volume_file_name, dtype=np.int)

    # create a dictionary to store the supervoxel ids and their type
    labeled_supervoxel_dict = {}

    # update the dictionary
    for i in range(labeled_data.shape[0]):
        labeled_supervoxel_dict[labeled_volume_data[labeled_data[i]]] = labeled_id[i]

    # update the node type.
    for key, value in labeled_supervoxel_dict.items():
        G.add_node(key, cls=str(value))
    # count the number
    ls = list(labeled_supervoxel_dict.values())
    for i in range(label_type_number):
        print("Type id : {} --- Supervoxel Number : {}".format(i, ls.count(i)))

    return label_type_number

# label graph for only one npy file
# def addLabelForGraph(G, label_npy_file, labeled_volume_file_name: str):
#
#
#
#     return label_type_number

if __name__ == "__main__":
    start = time.clock()

    hp = parse_args("SupervoxelGraph --- label graph")

    workspace = hp.workspace
    gexf_file = os.path.join(workspace, hp.graph_file)
    labeled_gexf_file = os.path.join(workspace, hp.labeled_graph_file)


    print('Begin to transform the unlabeled graph to labeled graph...')
    G = nx.read_gexf(gexf_file)

    if hp.type == 1:
        print('Loading combined csv labeled file...')
        labeled_voxel_file = workspace + hp.labeled_file
        labeled_int_volume_file = workspace + hp.label_file
        if os.path.exists(labeled_voxel_file):
            label_type_number = addLabelForGraph(G, labeled_voxel_file, labeled_int_volume_file)
            print('The number of labeled type: {}'.format(label_type_number))
    elif hp.type == 0:
        print('Loading combined csv labeled file from itk SNAP tool...')
        itk_snap_labeled_voxel_file = workspace + hp.labeled_file
        labeled_int_volume_file = workspace + hp.label_file
        if os.path.exists(itk_snap_labeled_voxel_file):
            label_type_number = addLabelForGraph(G, itk_snap_labeled_voxel_file, labeled_int_volume_file)
            print('The number of labeled type: {}'.format(label_type_number))
    # test sphere/lung file
    elif hp.type == 2:
        print('Loading ground_truth labeled file...')
        ground_truth_labeled_supervoxel_file = workspace + hp.labeled_file
        print(ground_truth_labeled_supervoxel_file)
        # labeled_int_volume_file = file_prefix + json_content["file_name"]["label_file"]
        if os.path.exists(ground_truth_labeled_supervoxel_file):

            # load labeled int raw file
            labeled_volume_data = np.fromfile(ground_truth_labeled_supervoxel_file, dtype=np.uint8).flatten()
            supervoxel_id_array = np.fromfile(workspace+hp.supervoxel_id_file, dtype=np.int32).flatten()
            # create a dictionary to store the supervoxel ids and their type
            labeled_supervoxel_dict = {}

            # update the dictionary
            for i in range(supervoxel_id_array.shape[0]):
                if supervoxel_id_array[i] in labeled_supervoxel_dict.keys():
                    labeled_supervoxel_dict[supervoxel_id_array[i]].append(labeled_volume_data[i])
                else:
                    labeled_supervoxel_dict[supervoxel_id_array[i]] = []

            # count the ambiguous super-voxels
            ambiguous_supervoxel_number = 0

            # update the node type.
            for key, value in labeled_supervoxel_dict.items():
                v = np.argmax(np.bincount(np.array(value)))
                # print(key, value, v)
                if len(set(value)) > 1:
                    ambiguous_supervoxel_number += 1
                labeled_supervoxel_dict[key] = v
                G.add_node(key, cls=str(v))

            # count the number
            ls = list(labeled_supervoxel_dict.values())
            label_type_number = len(set(labeled_supervoxel_dict.values()))
            for i in range(label_type_number):
                print("Type id : {}, supervoxel Number : {}".format(i, ls.count(i)))
            print("Ambiguous supervoxel number : {}, proportion : {:.2f}%".format(ambiguous_supervoxel_number,
                                                                                  ambiguous_supervoxel_number*100/len(ls)))
        else:
            print("Error")

            # # load ground truth npy file
            # labeled_id = np.load(ground_truth_labeled_supervoxel_file)
            # labeled_id = labeled_id.flatten()
            # for i in range(labeled_id.shape[0]):
            #     G.add_node(i, cls=str(labeled_id[i]))
            # print('The number of labeled type: {}'.format(len(set(labeled_id))))
    else:
        print('Loading separated csv labeled file...')
        for i in range(len(hp.csv_file)):
            if os.path.exists(hp.csv_file[i]):
                addLabelForGraph(G, hp.csv_file[i], i)
        print('The number of labeled type: %d' % len(hp.csv_file))

    saveGraph(G, labeled_gexf_file)
    elapsed = (time.clock() - start)
    print("Time for labelling graph: ", elapsed, "s.")
