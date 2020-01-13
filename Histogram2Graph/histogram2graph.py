import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

import os

def histogramToGraph(json_file_name):
    f = open(json_file_name)
    json_content = json.load(f)
    file_prefix = json_content["data_path"]["file_prefix"]
    labeled_histogram_file = json_content["file_name"]["label_histogram_file"]
    # label_histogram_array = np.loadtxt("label_histogram_array.txt")  # 缺省按照'%.18e'格式保存数据，以空格分隔
    label_histogram_array = np.loadtxt(file_prefix + labeled_histogram_file)  # 缺省按照'%.18e'格式保存数据，以空格分隔
    label_number = label_histogram_array.shape[0]

    edge_weight_file = json_content["file_name"]["edge_weight_file"]
    # edge_array = np.loadtxt("edge_weight_array.txt")
    edge_array = np.loadtxt(file_prefix + edge_weight_file)
    print("label number: ", label_number)
    print("edge information: ", edge_array.shape)

    G = nx.Graph()
    for i in range(label_number):
        G.add_node(i, cls=-1)

        for j in range(i + 1, label_number):
            if edge_array[i][j] > 0:
                G.add_edge(i, j)

    for i in range(label_number):
        for j in range(label_histogram_array[i].shape[0]):
            G.node[i][j] = str(label_histogram_array[i][j])

    gexf_file = file_prefix + json_content["file_name"]["graph_file"]

    return G, gexf_file

import csv
# 读取labeled文件
def loadLabelId(csv_file_name):
    labeled_data = []
    with open(csv_file_name) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        labeled_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            labeled_data.append(row[0])

    labeled_data = [int(x) for x in labeled_data]  # 将数据从string形式转换为float形式
    labeled_data = np.array(labeled_data)  # 将list数组转化成array数组便于查看数据结构
    return labeled_data

def addLabelForGraph(G, label_csv_file, label_id):
    labeled_data = loadLabelId(label_csv_file)
    # test_label_data = loadLabelId("labeled_test_outter_super_voxels.csv")
    for x in labeled_data:
        G.add_node(x, cls=label_id)

def saveGraph(G, graph_file_name):
    nx.write_gexf(G, graph_file_name)
    print("SuperGraph has been saved.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please use command line parameter.")
    else:
        G, gexf_file_name = histogramToGraph(sys.argv[1])

        for i in range(2, len(sys.argv)):
            if os.path.exists(sys.argv[i]):
                addLabelForGraph(G, sys.argv[i], i-1)

        saveGraph(G, gexf_file_name)

