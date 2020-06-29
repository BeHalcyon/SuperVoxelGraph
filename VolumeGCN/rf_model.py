# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:38:18 2018

@author: aoanng
"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import sys
import os
import numpy as np
import time
from args import parse_args
from module import Graphs, metricMeasure
sys.path.append("/")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main():
    time_start = time.time()

    # 1. read graph data
    hp = parse_args()
    G = Graphs(hp)
    # random label
    print("The gexf graph data has been loaded.")
    node_num = len(G.nodes())
    hp.node_num = node_num

    x = []
    y = []

    # 2. allocate data
    # all nodes feature vector
    print('The dimension of each node : {}'.format(hp.vec_dim))
    f_init = np.zeros((hp.node_num, hp.vec_dim), dtype=np.float32)

    for n in range(node_num):
        for i in range(hp.vec_dim):
            f_init[n][i] = G.node[str(n)][str(i)]
        if G.node[str(n)]['cls'] != -1:
            x.append(list(f_init[n]))
            y.append(G.node[str(n)]['cls'])

    hp.label = len(set(y))
    hp.labeled_node = len(y)
    print('Number of all nodes : ', hp.node_num)
    print('Number of labeled nodes : ', hp.labeled_node)
    print('Number of trained labeled nodes : ', int(hp.labeled_node * hp.ratio))
    print('Number of test labeled nodes : ', int(hp.labeled_node * (1 - hp.ratio)))
    x = np.array(x)
    y = np.array(y, dtype=np.int)

    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=hp.ratio,
                                                                      test_size=1-hp.ratio)  # sklearn.model_selection.
    train_data = np.array(x)
    train_label = np.array(y)

    print(train_data.shape)
    print(train_label.shape)

    # 随机森林
    clf = RandomForestClassifier(n_estimators=40, max_features=0.1, criterion='entropy', max_depth=None, min_samples_split=2,
                                  random_state=0)

    cross_val_scores = cross_val_score(clf, train_data, train_label)
    print("Cross value score for tree number={} and feature number={} is : {}".
          format(40, 0.1, cross_val_scores.mean()))
    # iris_data = load_iris()
    # train_data = iris_data.data
    # train_label = iris_data.target

    # parameterSelect(train_data, train_label)

    clf.fit(train_data, train_label)
    train_score = clf.score(train_data, train_label)
    test_score = clf.score(test_data, test_label)
    print("Train acc : {}".format(train_score))
    print("Test acc : {}".format(test_score))

    print("predict_result : ")
    predictions = clf.predict(test_data)
    print(predictions)

    print("truth_result : ")
    print(test_label)

    precision_sorce, recall_score, f1_score = metricMeasure(test_label, predictions)
    print("precision score : {}".format(precision_sorce))
    print("recall score : {}".format(recall_score))
    print("f1 score : {}".format(f1_score))

    print("All predict result : ")
    predictions = clf.predict(f_init)
    print(predictions)

    np.save(hp.predict_labeled_supervoxel_file, np.array(predictions))

    time_end = time.time()
    all_time = int(time_end - time_start)
    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print()
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')

def parameterSelect(train_data, train_label):
    for i in range(1, 100, 3):
        for j in range(1, 11):
            clf = RandomForestClassifier(n_estimators=i, max_features=j*0.1,
                                         criterion='entropy',
                                         max_depth=None,
                                         min_samples_split=2,
                                         random_state=0)
            # clf.fit(train_data, train_label)
            cross_val_scores = cross_val_score(clf, train_data, train_label)
            print("Cross value score for tree number={} and feature number={} is : {}".
                  format(i, j, cross_val_scores.mean()))


if __name__ == '__main__':
    main()
