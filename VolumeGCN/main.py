# -*- coding: utf-8 -*-
# /usr/bin/python3
import sys

sys.path.append("/")
import tensorflow as tf
from model import Volume_GCN
from tqdm import tqdm
from module import Graphs
from load_data import train_data
from sklearn import metrics
from args import parse_args
import math
import time
import random

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main():
    time_start = time.time()

    hp = parse_args()
    # print("开始读取数据")

    G = Graphs(hp)
    # random label
    print("The gexf graph data has been loaded.")

    node_num = len(G.nodes())
    hp.node_num = node_num
    # labeled_nodes = [[i, i%3] for i in range(300)]
    # labeled_nodes格式：[[node_id, node_cls],[],[],...]，node_id表示这个节点的新的id，node_cls表示类别
    labeled_nodes = []
    all_nodes = []
    label_set = set()
    for n in range(node_num):
        all_nodes.append(n)
        if G.node[str(n)]['cls'] != -1:
            labeled_nodes.append([n, G.node[str(n)]['cls']])
            label_set.add(G.node[str(n)]['cls'])
    hp.label = len(label_set)
    random.shuffle(labeled_nodes) # 标签打散

    print('Number of labeled nodes : ', len(labeled_nodes))

    hp.labeled_node = len(labeled_nodes)
    # print(node_num)
    # print("读取数据完成，填入模型参数")

    arg = {}
    arg['hp'] = hp
    print("The model parameters have been set.")
    # print("构建模型")
    m = Volume_GCN(arg, G)
    A = tf.placeholder(dtype=tf.float32, shape=(node_num, node_num), name='A')
    # 训练集
    xs = tf.placeholder(dtype=tf.int32, shape=(int(hp.labeled_node * hp.ratio)), name='xs')
    # 训练集
    ys = tf.placeholder(dtype=tf.int32, shape=(int(hp.labeled_node * hp.ratio)), name='ys')
    # 测试集
    xu = tf.placeholder(dtype=tf.int32, shape=(hp.labeled_node - int(hp.labeled_node * hp.ratio)), name='xu')
    # 全部测试集
    xu_all = tf.placeholder(dtype=tf.int32, shape=(node_num), name='xu_all')


    loss, train_op, global_step = m.train(A, xs, ys)
    tf.summary.scalar('loss', loss)

    predict_label = m.predict(A, xu)
    predict_all_labels = m.predict(A, xu_all)
    dA, dxs, dys, dxu, dyu = train_data(hp, node_num, G, labeled_nodes)

    print("The model has been constructed. Starting to train...")

    # Save the training model
    saver = tf.train.Saver()  # generate saver

    config = tf.ConfigProto(allow_soft_placement=True)  # if the chosen device is not existed, tf can automatically distributes equipment
    config.gpu_options.allow_growth = True  # allocate memory dynamically

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        _gs = sess.run(global_step)

        from datetime import datetime
        import os
        cur_time = datetime.now().strftime("%Y%m%d%H%M%S")
        dir = os.path.join('./tensorboard/', cur_time)
        summary_writer = tf.summary.FileWriter(dir, graph=tf.get_default_graph())

        merge_summary = tf.summary.merge_all()  # 整合所有summary op

        for i in tqdm(range(hp.epochs)):
            _loss, _, _gs, s = sess.run([loss, train_op, global_step, merge_summary], feed_dict={A: dA, xs: dxs, ys: dys})

            print("   Epoch : %02d   loss : %.2f" % (i+1, _loss))

            summary_writer.add_summary(s, i)

        _pre = sess.run([predict_label], feed_dict={A: dA, xu: dxu})

        print("Fin accuracy is : ", metrics.accuracy_score(dyu, _pre[0]))

        # predict all nodes.
        _pre2 = sess.run([predict_all_labels], feed_dict={A: dA, xu_all: all_nodes})
        print(_pre2[0])

        import numpy as np
        ground_truth_array = np.load(hp.ground_truth_labeled_supervoxel_file)
        print("All accuracy is : ", metrics.accuracy_score(ground_truth_array, _pre2[0]))

        np.save(hp.predict_labeled_supervoxel_file, np.array(_pre2[0]))

        saver.save(sess, "model/"+cur_time)


        # print("Fin AUC score is : ", metrics.auc(dyu, _pre[0])) # tf不支持多分类，得另外写
    time_end = time.time()
    all_time = int(time_end - time_start)



    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')


if __name__ == '__main__':
    main()