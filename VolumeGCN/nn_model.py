# -*- coding:utf-8 -*-
"""
@author:hxy
@file:nn_model.py
@func:Use simple neural network to achieve tooth classification
@time:2020/6/24
"""
from keras.utils import np_utils
import numpy as np
import time
from args import parse_args
import sys
import os
from module import Graphs, metricMeasure
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import metrics

sys.path.append("/")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


from keras import backend as K



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

    # train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=1,
    #                                                                   test_size=0)  # sklearn.model_selection.
    train_data = np.array(x)
    train_label = np.array(y)

    print(train_data.shape)
    # print(test_data.shape)
    print(train_label.shape)
    # print(test_label.shape)

    predictions = nnModel(hp, f_init, train_data, train_label)

    np.save(hp.predict_labeled_supervoxel_file, np.array(predictions))

    time_end = time.time()
    all_time = int(time_end - time_start)

    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print()
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')


def nnModel(hp, f_init, train_data, train_label):
    y_train_onehot = np_utils.to_categorical(train_label)
    # y_test_onehot = np_utils.to_categorical(test_label)
    # 声明序贯模型
    model = Sequential()
    model.add(Dense(units=hp.hidden1,
                    input_dim=hp.vec_dim,
                    kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(units=hp.hidden2,
                    kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(units=hp.label,
                    kernel_initializer='normal',
                    activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    train_history = model.fit(x=train_data,
                              y=y_train_onehot,
                              validation_split=1 - hp.ratio,
                              epochs=100,
                              batch_size=10,
                              verbose=2)
    # train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=hp.ratio,
    #                                                 test_size=1-hp.ratio)  # sklearn.model_selection.
    # 预测样本类别
    predictions = model.predict_classes(train_data)
    print("predict_result : ")
    print(predictions)
    print("truth_result : ")
    print(train_label)
    precision_sorce, recall_score, f1_score = metricMeasure(train_label, predictions)
    print("precision score : {}".format(precision_sorce))
    print("recall score : {}".format(recall_score))
    print("f1 score : {}".format(f1_score))
    print("All predict result : ")
    predictions = model.predict_classes(f_init)
    print(predictions)
    return predictions


if __name__ == '__main__':
    main()

#
# np.random.seed(10)
#
# from keras.datasets import mnist
# (x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
#
# x_train = x_train_image.reshape(60000, 784).astype('float32')
# x_test = x_test_image.reshape(10000, 784).astype('float32')
#
# x_train_norm = x_train /255
# x_test_norm = x_test /255
#
# y_train_onehot = np_utils.to_categorical(y_train_label)
# y_test_onehot = np_utils.to_categorical(y_test_label)
#
# from keras.models import Sequential
# from keras.layers import Dense
#
# # 声明序贯模型
# model = Sequential()
#
# model.add(Dense(units=256,
#                 input_dim=784,
#                 kernel_initializer='normal',
#                 activation='relu'))
#
# model.add(Dense(units=10,
#                 kernel_initializer='normal',
#                 activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam' ,metrics=['accuracy'])
#
# train_history = model.fit(x=x_train_norm,
#                         y=y_train_onehot,
#                         validation_split=0.2,
#                         epochs=10,
#                         batch_size=200,
#                         verbose=2)
