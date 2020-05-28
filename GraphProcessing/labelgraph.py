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


def parse_args():
    parser = argparse.ArgumentParser(description="Histogram to Graph")

    parser.add_argument('--configure_file', default='../x64/Release/workspace/spheres_supervoxel.json',
                        help='configure json file')
    parser.add_argument('--csv_file', nargs='+', help='csv file export by gephi')

    args = parser.parse_args()
    return args


def saveGraph(G, graph_file_name):
    nx.write_gexf(G, graph_file_name)
    print("Labeled SuperGraph has been saved.")


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


if __name__ == "__main__":
    start = time.clock()

    hp = parse_args()
    f = open(hp.configure_file)
    json_content = json.load(f)

    file_prefix = json_content["data_path"]["file_prefix"]
    gexf_file = file_prefix + json_content["file_name"]["graph_file"]
    labeled_gexf_file = file_prefix + json_content["file_name"]["labeled_graph_file"]

    print('Begin to transform the unlabeled graph to labeled graph...')

    G = nx.read_gexf(gexf_file)

    # print(hp)
    # saveGraph(G, gexf_file_name)
    for i in range(len(hp.csv_file)):
        if os.path.exists(hp.csv_file[i]):
            addLabelForGraph(G, hp.csv_file[i], i)

    print('The number of labeled type: %d' % len(hp.csv_file))

    saveGraph(G, labeled_gexf_file)

    elapsed = (time.clock() - start)
    print("Time for labelling graph: ", elapsed, "s.")
