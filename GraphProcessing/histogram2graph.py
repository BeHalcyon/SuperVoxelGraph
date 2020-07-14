
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

import os
import time
from sklearn import preprocessing

def histogramToGraph(json_file_name):
    f = open(json_file_name)
    json_content = json.load(f)
    file_prefix = json_content["data_path"]["file_prefix"]
    labeled_histogram_file = json_content["file_name"]["label_histogram_file"]
    label_histogram_array = np.loadtxt(file_prefix + labeled_histogram_file)  # 缺省按照'%.18e'格式保存数据，以空格分隔
    label_number = label_histogram_array.shape[0]

    edge_weight_file = json_content["file_name"]["edge_weight_file"]
    # edge_array = np.loadtxt("edge_weight_array.txt")
    edge_array = np.loadtxt(file_prefix + edge_weight_file)
    print("label number: ", label_number)
    print("edge information: ", edge_array.shape)

    # print(label_histogram_array[:, 127:])

    # normalize feature
    scalar = preprocessing.StandardScaler().fit(label_histogram_array)
    # print(scaler.mean_)
    # print(scaler.var_)
    label_histogram_array = scalar.transform(label_histogram_array)
    # print(f_init[:10])
    # print(res[:10])
    # print(label_histogram_array[:, 127:])
    #
    print('Node number : {} Feature number : {}'.format(label_histogram_array.shape[0],
                                                        label_histogram_array.shape[1]))
    # return


    G = nx.Graph()

    for i in range(label_number):
        G.add_node(i, cls=-1)
        for j in range(label_histogram_array[i].shape[0]):
            G.node[i][j] = str(label_histogram_array[i][j])

    # for i in range(label_number):
    #     G.add_node(i, cls=-1)
        # for j in range(i + 1, label_number):
        #     if edge_array[i][j] > 0:
        #         G.add_edge(i, j)
    # Add the edge information

    # Extension 20200709 add weight to edge

    for j in edge_array:
        if j[2] != 0:
            G.add_edge(int(j[0]), int(j[1]), weight=1)
            # G.add_edge(int(j[0]), int(j[1]))


    # print(G.node)

    gexf_file = file_prefix + json_content["file_name"]["graph_file"]

    return G, gexf_file


def saveGraph(G, graph_file_name):
    nx.write_gexf(G, graph_file_name)
    print("SuperGraph has been saved.")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Histogram to Graph")

    parser.add_argument('--configure_file', default='../x64/Release/workspace/spheres_supervoxel.json',
                        help='configure json file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    start = time.clock()
    hp = parse_args()

    print('Begin to transform the vector information to unlabeled graph...')

    G, gexf_file_name = histogramToGraph(hp.configure_file)

    # for i in range(2, len(sys.argv)):
    #     if os.path.exists(sys.argv[i]):
    #         addLabelForGraph(G, sys.argv[i], i-1)


    # print(G.node[0])

    saveGraph(G, gexf_file_name)

    elapsed = (time.clock() - start)
    print("Time for histogram2graph: ", elapsed, "s.")
