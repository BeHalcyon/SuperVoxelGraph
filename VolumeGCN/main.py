# -*- coding: utf-8 -*-
# /usr/bin/python3
import sys

sys.path.append("/")
import tensorflow as tf
from model import Volume_GCN
from tqdm import tqdm
from module import Graphs, metricMeasure
from load_data import train_data
from sklearn import metrics
from args import parse_args
import math
import time
import random
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import random


# 从列表中随机选取一定数目的元素
def random_sample(num1, num2):
    dataList = list(range(num1))  # 产生指定范围元素的列表
    TrainIndex = []  # 初始化存储随机产生的索引的列表
    num2 = num1 if num2 > num1 else num2
    for i in range(num2):  # 随机产生索引的数量
        randIndex = int(random.uniform(0, len(dataList)))  # 在指定的范围内产生随机数位置索引
        TrainIndex.append(dataList[randIndex])  # 进行存all_nodes储
        del (dataList[randIndex])  # 对已选中的元素进行删除，以便于下一次随机选取
    TestIndex = dataList  # 随机选取过后剩下的元素
    return TrainIndex, TestIndex  # 返回随机选取的一定数目的元素，和剩下的元素


def main():
    import numpy as np
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

    is_ground_truth_data_used = True
    ground_truth_array = []

    if hp.labeled_type != 2:
        for n in range(node_num):
            all_nodes.append(n)
            if G.nodes[str(n)]['cls'] != -1:
                labeled_nodes.append([n, G.nodes[str(n)]['cls']])
                label_set.add(G.nodes[str(n)]['cls'])
        hp.label = len(label_set)
        random.shuffle(labeled_nodes)  # 标签打散
        # labeled_nodes = labeled_nodes[:2000]
    # Using ground truth training set
    else:

        # ground_truth_array stores the supervoxels' id. 1D array [id1, id2, id3, ..., ]
        import numpy as np
        for n in range(node_num):
            all_nodes.append(n)
            ground_truth_array.append(int(G.nodes[str(n)]['cls']))
        ground_truth_array = np.array(ground_truth_array)
        # np.save(hp.groundtruth_label_supervoxel_file, ground_truth_array)
        all_nodes_number = node_num

        all_nodes = [n for n in range(node_num)]
        train_set_number = int(all_nodes_number * 0.2)

        type_number = int(ground_truth_array.max()) + 1
        # the number of train set for each type, sample average
        train_set_number_for_each_type = train_set_number // type_number
        train_set_number_for_each_type_list = [train_set_number_for_each_type] * type_number

        train_set_number_for_each_type_list = [int(train_set_number * np.sum(ground_truth_array == i) / len(
            ground_truth_array)) for i in range(type_number)]

        for i in range(type_number):
            buf_array = ground_truth_array[ground_truth_array == i]
            buf_array_index = np.array(np.where(ground_truth_array == i)[0])
            train_index_array, _ = random_sample(buf_array.shape[0], train_set_number_for_each_type_list[i])

            for n in train_index_array:
                labeled_nodes.append([buf_array_index[n], i])
                # print([buf_array_index[n], i])
                label_set.add(i)
        hp.label = len(label_set)
        random.shuffle(labeled_nodes)  # 标签打散

        # print(labeled_nodes)

    hp.labeled_node = len(labeled_nodes)
    print('Number of all nodes : ', hp.node_num)
    print('Number of labeled nodes : ', hp.labeled_node)
    print('Number of trained labeled nodes : ', int(hp.labeled_node * hp.ratio))
    print('Number of test labeled nodes : ', int(hp.labeled_node * (1 - hp.ratio)))


    # print(node_num)
    # print("读取数据完成，填入模型参数")

    arg = {}
    arg['hp'] = hp
    print("The model parameters have been set.")
    # print("构建模型")

    # initial feature vector
    initial_feature_vector = np.load(hp.workspace + hp.graph_node_feature_file)

    assert initial_feature_vector.shape[0] == hp.node_num

    initial_feature_vector = initial_feature_vector[:, :hp.vec_dim]

    m = Volume_GCN(arg, G, initial_feature_vector)
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
    dA, dxs, dys, dxu, dyu = train_data(hp, node_num, G, labeled_nodes)
    time2 = time.time()

    print("Laplacian matrix calculation time : {} second.".format(int(time2 - time1)))
    print("=============================================================")
    print("Start training...")
    print("=============================================================")

    # Save the training model
    saver = tf.train.Saver()  # generate saver

    config = tf.ConfigProto(
        allow_soft_placement=True)  # if the chosen device is not existed, tf can automatically distributes equipment
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
            _loss, _, _gs, s = sess.run([loss, train_op, global_step, merge_summary],
                                        feed_dict={A: dA, xs: dxs, ys: dys})
            print("   Epoch : %02d   loss : %.4f" % (i + 1, _loss))

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

        # ground truth
        if hp.labeled_type == 2:
            # ground_truth_array = np.load(hp.ground_truth_labeled_supervoxel_file)
            pre_results = _pre2[0]
            filter_ground_truth_array = ground_truth_array[ground_truth_array != -1]
            pre_results = pre_results[ground_truth_array != -1]
            # print("All accuracy is : ", metrics.accuracy_score(ground_truth_array, _pre2[0]))
            # TODO
            print("All accuracy is : ", metrics.accuracy_score(filter_ground_truth_array, pre_results))

            # voxel accuracy:
            # TODO: in supervoxel2voxel.py

        saver.save(sess, "model/" + cur_time)

        # print("Fin AUC score is : ", metrics.auc(dyu, _pre[0])) # tf不支持多分类，得另外写
    time_end = time.time()
    all_time = int(time_end - time_start)

    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')


if __name__ == '__main__':
    main()
