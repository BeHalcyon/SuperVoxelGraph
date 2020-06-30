# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
    # TODO : Test the model ?
    train_data = np.array(x)
    train_label = np.array(y)

    print(train_data.shape)
    # print(test_data.shape)
    print(train_label.shape)
    # print(test_label.shape)

    # parameterSelect(test_data, test_label, train_data, train_label)

    predictions2 = knnModel(f_init, train_data, train_label, test_data, test_label)

    np.save(hp.predict_labeled_supervoxel_file, np.array(predictions2))


    time_end = time.time()
    all_time = int(time_end - time_start)

    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print()
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')


def knnModel(f_init, train_data, train_label, test_data, test_label):
    knn = KNeighborsClassifier(1, p=2)
    knn.fit(train_data, train_label)
    a = knn.score(train_data, train_label)
    b = knn.score(test_data, test_label)
    print("predict_result : ")
    predictions = knn.predict(test_data)
    print(predictions)
    print("truth_result : ")
    print(test_label)
    precision_sorce, recall_score, f1_score = metricMeasure(test_label, predictions)
    print("precision score : {}".format(precision_sorce))
    print("recall score : {}".format(recall_score))
    print("f1 score : {}".format(f1_score))
    print("All predict result : ")
    predictions2 = knn.predict(f_init)
    print(predictions2)
    return predictions2


def parameterSelect(test_data, test_label, train_data, train_label):
    training_accuracy = []
    test_accuracy = []
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        knn = KNeighborsClassifier(n_neighbors, p=3)
        knn.fit(train_data, train_label)
        a = knn.score(train_data, train_label)

        training_accuracy.append(a)
        b = knn.score(test_data, test_label)
        test_accuracy.append(b)
        print('n_neighbors : {} . Accuracy for train : {}. Accuracy for test : {}'.format(n_neighbors, a, b))
    plt.plot(neighbors_settings, training_accuracy, label='Training Accuracy')
    plt.plot(neighbors_settings, test_accuracy, label='Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('n_neighbors')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

# cancer=load_iris()
#
# x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=66)
#
# print(x_train, y_train)
#
# training_accuracy=[]
# test_accuracy=[]
#
# neighbors_settings=range(1,11)
#
# for n_neighbors in neighbors_settings:
#     knn=KNeighborsClassifier(n_neighbors)
#     knn.fit(x_train, y_train)
#     training_accuracy.append(knn.score(x_train,y_train))
#     test_accuracy.append(knn.score(x_test,y_test))
#
# plt.plot(neighbors_settings,training_accuracy,label='Training Accuracy')
# plt.plot(neighbors_settings,test_accuracy,label='Test Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('n_neighbors')
# plt.legend()
# plt.show()