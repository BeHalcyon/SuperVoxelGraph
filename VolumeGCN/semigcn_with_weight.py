# -*- coding: utf-8 -*-
# /usr/bin/python3
import sys
sys.path.append("/")
import tensorflow as tf
from model import Volume_GCN
from tqdm import tqdm
from module import Graphs, metricMeasure
from load_data import train_data, train_data_with_weight
from sklearn import metrics
from args import parse_args
import time

import os
from sklearn.decomposition import PCA

sys.path.append("/")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import random


# 从列表中随机选取一定数目的元素
def random_sample(num1, num2):
    dataList = list(range(num1))  # 产生指定范围元素的列表
    TrainIndex = []  # 初始化存储随机产生的索引的列表
    num2 = num1 if num2 > num1 else num2
    for i in range(num2):  # 随机产生索引的数量
        randIndex = int(random.uniform(0, len(dataList)))  # 在指定的范围内产生随机数位置索引
        TrainIndex.append(dataList[randIndex])  # 进行存储
        del (dataList[randIndex])  # 对已选中的元素进行删除，以便于下一次随机选取
    TestIndex = dataList  # 随机选取过后剩下的元素
    return TrainIndex, TestIndex  # 返回随机选取的一定数目的元素，和剩下的元素

def main():
    time_start = time.time()
    hp = parse_args()
    # print("开始读取数据")

    G = Graphs(hp)
    # random label
    print("The gexf graph data has been loaded.")

    node_num = len(G.nodes())
    hp.node_num = node_num

    x = []
    y = []

    # allocate data
    # all nodes feature vector
    print('The dimension of each node : {}'.format(hp.vec_dim))

    import numpy as np
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

    # hp.vec_dim = pca.n_components_

    gradients = f_init[:, -5]
    max_gradient = np.max(gradients)
    min_gradient = np.min(gradients)
    gradient_norm = 1/(max_gradient-min_gradient)

    all_nodes = []
    labeled_nodes = []
    for n in range(node_num):
        all_nodes.append(n)
        # for i in range(hp.vec_dim):
        #     G.node[str(n)][str(i)] = new_x[n, i]
        if G.node[str(n)]['cls'] != -1:
            labeled_nodes.append([n, G.node[str(n)]['cls']])
    for edge in G.edges():
        weight = 1 - gradient_norm*abs(gradients[int(edge[0])] - gradients[int(edge[1])])
        G[edge[0]][edge[1]]['weight'] = weight
        G[edge[1]][edge[0]]['weight'] = weight
        # print(gradient_norm, abs(gradients[int(edge[0])] - gradients[int(edge[1])]), weight)


    random.shuffle(labeled_nodes)  # 标签打散

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

    print("=============================================================")
    print("The model has been constructed. Start calculating laplacian matrix...")
    time1 = time.time()
    dA, dxs, dys, dxu, dyu = train_data_with_weight(hp, node_num, G, labeled_nodes)
    time2 = time.time()

    print("Laplacian matrix calculation time : {} second.".format(int(time2-time1)))
    print("=============================================================")
    print("Start training...")
    print("=============================================================")

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

            print("   Epoch : %02d   loss : %.4f" % (i+1, _loss))

            summary_writer.add_summary(s, i)

        _pre = sess.run([predict_label], feed_dict={A: dA, xu: dxu})

        print("Fin accuracy is : ", metrics.accuracy_score(dyu, _pre[0]))
        print("predict_result : ")
        print(_pre[0])
        print("truth_result : ")
        print(dyu)

        precision_sorce, recall_score, f1_score = metricMeasure(dyu, _pre[0])
        print("precision score : {}".format(precision_sorce))
        print("recall score : {}".format(recall_score))
        print("f1 score : {}".format(f1_score))

        # # predict all nodes.
        # _pre2 = sess.run([predict_all_labels], feed_dict={A: dA, xu_all: all_nodes})
        # print(_pre2[0])

        import numpy as np

        # predict model for multiple types of nodes.
        # predict all nodes.
        _pre2 = sess.run([predict_all_labels], feed_dict={A: dA, xu_all: all_nodes})
        print("All predict result : ")
        print(_pre2[0])
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