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

def loadNiiFile(file_name):
    import SimpleITK
    itk_img = SimpleITK.ReadImage(file_name)
    img_array = SimpleITK.GetArrayFromImage(itk_img)  # the array is arranged with [z, y, x]
    # print(img_array.dtype, img_array.shape)
    dimension = itk_img.GetSize()  # the dimension is arranged with [x, y, z]
    origin = itk_img.GetOrigin()
    direction = itk_img.GetDirection()
    space = itk_img.GetSpacing()

    return img_array.flatten().astype(np.int32)

if __name__ == "__main__":
    start = time.clock()

    hp = parse_args("SupervoxelGraph --- label graph")

    workspace = hp.workspace
    gexf_file = os.path.join(workspace, hp.graph_file)
    labeled_gexf_file = os.path.join(workspace, hp.labeled_graph_file)


    print('Begin to transform the unlabeled graph to labeled graph...')
    G = nx.read_gexf(gexf_file)

    # print(G.nodes)
    # print(G.nodes[str(100)])


    if hp.type == 1:
        print('Loading labeled nii.gz/nii file from ITK-SNAP...')
        labeled_voxel_file = workspace + hp.labeled_file
        # Warning: one should label volume in itk-snap and export the mask nii file, then update the json configure file.
        labeled_voxel_array = loadNiiFile(labeled_voxel_file)
        # 0 is unlabeled voxel. The value larger than 0 is available.
        print(max(labeled_voxel_array), min(labeled_voxel_array))
        supervoxel_id_array = np.fromfile(workspace + hp.supervoxel_id_file, dtype=np.int32).flatten()
        assert max(supervoxel_id_array) + 1 == len(G.nodes())
        assert supervoxel_id_array.shape == labeled_voxel_array.shape
        # create a dictionary to store the supervoxel ids and their type
        labeled_supervoxel_dict = {}
        # update the dictionary
        for i in range(supervoxel_id_array.shape[0]):
            if labeled_voxel_array[i] != 0:
                if supervoxel_id_array[i] in labeled_supervoxel_dict.keys():
                    labeled_supervoxel_dict[supervoxel_id_array[i]].append(labeled_voxel_array[i]-1)
                else:
                    labeled_supervoxel_dict[supervoxel_id_array[i]] = [labeled_voxel_array[i]-1]

        for i in range(len(G.nodes())):
            if i not in labeled_supervoxel_dict.keys():
                labeled_supervoxel_dict[i] = [-1]
        # count the ambiguous super-voxels
        ambiguous_supervoxel_number = 0
        ground_truth_supervoxel_label_array = np.zeros(len(labeled_supervoxel_dict.keys()), dtype=np.int32)
        # update the node type.
        for key, value in labeled_supervoxel_dict.items():
            v = -1
            # print(key, value, v)
            if len(set(value)) > 1:
                buf = np.bincount(value) / len(value)
                for i in range(len(buf)):
                    if buf[i] >= 0.5:
                        v = i
                        continue
                if v == -1:
                    ambiguous_supervoxel_number += 1
            else:
                v = value[0]
            labeled_supervoxel_dict[key] = v
            G.nodes[str(key)]['cls'] = v
            ground_truth_supervoxel_label_array[key] = v

        np.save(hp.workspace + hp.groundtruth_label_supervoxel_file, ground_truth_supervoxel_label_array)

        # count the number
        ls = list(labeled_supervoxel_dict.values())
        ls2 = list(labeled_voxel_array)
        # label_type_set = set(labeled_supervoxel_dict.values())
        for i in set(labeled_supervoxel_dict.values()):
            print("Type id : {}, supervoxel Number : {}, voxel_number : {}".format(i, ls.count(i), ls2.count(i+1)))
        print("labeled supervoxel proportion : {:.2f}%".format((1-(ls.count(-1))/len(ls))*100))
        print("labeled voxel proportion : {:.2f}%".format((1-(ls2.count(0))/len(ls2))*100))
        print("Ambiguous supervoxel number : {}, proportion : {:.2f}%".format(ambiguous_supervoxel_number,
                                                                              ambiguous_supervoxel_number * 100 / len(
                                                                                  ls)))
        print("Average number of voxels in a supervoxel : {:.2f}".format(
            len(labeled_voxel_array) / len(labeled_supervoxel_dict)))

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
            print(max(labeled_volume_data))
            supervoxel_id_array = np.fromfile(workspace+hp.supervoxel_id_file, dtype=np.int32).flatten()
            assert max(supervoxel_id_array)+1 == len(G.nodes())

            # create a dictionary to store the supervoxel ids and their type
            labeled_supervoxel_dict = {}


            # update the dictionary
            for i in range(supervoxel_id_array.shape[0]):
                if supervoxel_id_array[i] in labeled_supervoxel_dict.keys():
                    labeled_supervoxel_dict[supervoxel_id_array[i]].append(labeled_volume_data[i])
                else:
                    labeled_supervoxel_dict[supervoxel_id_array[i]] = [labeled_volume_data[i]]

            # new_array = np.zeros(supervoxel_id_array.shape, dtype=np.uint8)
            # for key, value in labeled_supervoxel_dict.items():
            #     new_array[supervoxel_id_array == key] = value[0]
            #
            #
            # new_array.tofile("test_supervoxel_volume.raw")
            # print("test---------------------------------------")


            # count the ambiguous super-voxels
            ambiguous_supervoxel_number = 0
            voxel_number_in_supervoxel_list = []
            ground_truth_supervoxel_label_array = np.zeros(len(labeled_supervoxel_dict.keys()), dtype=np.int32)
            # update the node type.
            for key, value in labeled_supervoxel_dict.items():
                v = -1
                # print(key, value, v)
                if len(set(value)) > 1:
                    buf = np.bincount(value)/len(value)
                    for i in range(len(buf)):
                        if buf[i] >= 0.5:
                            v = i
                            continue
                    if v == -1:
                        ambiguous_supervoxel_number += 1
                else:
                    v = value[0]
                labeled_supervoxel_dict[key] = v
                voxel_number_in_supervoxel_list.append(len(value))
                # G.add_node(key, cls=v)
                G.nodes[str(key)]['cls'] = v
                ground_truth_supervoxel_label_array[key] = v

            np.save(hp.workspace + hp.groundtruth_label_supervoxel_file, ground_truth_supervoxel_label_array)

            # count the number
            ls = list(labeled_supervoxel_dict.values())
            # label_type_set = set(labeled_supervoxel_dict.values())
            for i in set(labeled_supervoxel_dict.values()):
                print("Type id : {}, supervoxel Number : {}".format(i, ls.count(i)))
            print("Ambiguous supervoxel number : {}, proportion : {:.2f}%".format(ambiguous_supervoxel_number,
                                                                                  ambiguous_supervoxel_number*100/len(ls)))
            print("Average number of voxels in a supervoxel : {:.2f}".format(sum(voxel_number_in_supervoxel_list)/len(voxel_number_in_supervoxel_list)))
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
