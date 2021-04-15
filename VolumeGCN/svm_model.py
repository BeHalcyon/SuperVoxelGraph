# -*- coding:utf-8 -*-
"""
@author:hxy
@file:svm_model.py
@func:Use SVM to achieve tooth classification
@time:2020/6/24
"""
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import time
from args import parse_args
import sys
import os
from module import Graphs, metricMeasure
import random
import numpy as np
from module import get_initialization
from sklearn.model_selection import cross_val_score

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
    print(y)
    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=hp.ratio,
                                                                      test_size=1 - hp.ratio)  # sklearn.model_selection.

    predict_result = svmModel(f_init, train_data, test_label, test_data, train_label)
    np.save(hp.predict_labeled_supervoxel_file, np.array(predict_result))

    print('The predict labeled supervoxel file trained by svm has been save in : {}'.
          format(hp.predict_labeled_supervoxel_file))

    time_end = time.time()
    all_time = int(time_end - time_start)

    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')


def predictSVM(index, process_number, f_init, classifier):

    feature_number = f_init.shape[0]
    length = (feature_number + process_number - 1) // process_number
    start_index = length * index
    end_index = length * (index + 1)
    if end_index > feature_number:
        end_index = feature_number

    batch_size = 4196
    batch_number = (end_index - start_index + batch_size - 1) // batch_size
    predict_result = np.zeros(end_index - start_index, dtype=np.int32)
    for i in range(batch_number):
        if i % 10 == 0:
            print('Thread id: {}, Processing svm prediction: {:.2f}%'.format(index, i * 100 / (batch_number - 1)))
        end_pos = start_index + (i + 1) * batch_size
        if end_pos > end_index:
            end_pos = end_index
        predict_result[i * batch_size:end_pos - start_index] = classifier.predict(
            f_init[start_index + i * batch_size:end_pos, :])

    return predict_result


def svmModel(f_init, train_data, test_label, test_data, train_label):
    import time
    time_start = time.time()
    # 3. train svm
    C = 1
    gamma = 0.03
    # max_acc = 0
    # parameter_ls = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    # for i in range(len(parameter_ls)):
    #     for j in range(len(parameter_ls)):
    #         classifier = svm.SVC(C=parameter_ls[i], kernel='rbf', gamma=parameter_ls[j],
    #                              decision_function_shape='ovr')  # ovr:一对多策略
    #         classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
    #         acc = classifier.score(test_data, test_label)
    #         if max_acc <= acc:
    #             C = parameter_ls[i]
    #             gamma = parameter_ls[j]
    #             max_acc = acc
    #             print('Update C to {} and gamma to {}. Current accurate: {}.'.format(C, gamma, acc))
    classifier = svm.SVC(C=C, kernel='rbf', gamma=gamma, decision_function_shape='ovr')  # ovr:一对多策略
    classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
    # 4.计算svc分类器的准确率
    # print("交叉验证：", cross_val_score(classifier, train_data, train_label))
    print("训练集：", classifier.score(train_data, train_label))
    print("测试集：", classifier.score(test_data, test_label))
    # 查看决策函数
    # print('train_decision_function:\n', classifier.decision_function(train_data))  # (90,3)
    predict_result = classifier.predict(test_data)
    print('predict_result:\n', predict_result)
    print('true_result:\n', test_label)
    precision_sorce, recall_score, f1_score = metricMeasure(test_label, predict_result)
    print("precision score : {}".format(precision_sorce))
    print("recall score : {}".format(recall_score))
    print("f1 score : {}".format(f1_score))

    time_end = time.time()
    print("Training time for svm : {}s".format(time_end - time_start))

    if f_init.shape[0] < 1e5:
        predict_result = classifier.predict(f_init)
        print("Predicting time for rf : {}s".format(time.time() - time_end))
        return predict_result.flatten().astype(np.int32)

    # # multiprocessing
    # import multiprocessing
    #
    # process_number = 8
    # pool = multiprocessing.Pool(processes=process_number)
    #
    #
    # results = []
    # for i in range(process_number):
    #     results.append(pool.apply_async(predictSVM, (i, process_number, f_init, classifier)))
    # pool.close()
    # pool.join()
    #
    # results = [res.get() for res in results]
    # predict_result = []
    # for res in results:
    #     for a in res:
    #         predict_result.append(a)
    # predict_result = np.array(predict_result, dtype=np.int32)
    #
    # batch_size = 4196
    # batch_number = len(f_init) // batch_size + 1
    # predict_result = np.zeros(f_init.shape[0], dtype=np.int32)

    # predict all results
    # batch prediction
    batch_size = 4196
    batch_number = (f_init.shape[0]+batch_size-1)//batch_size
    predict_result = np.zeros(f_init.shape[0], dtype=np.int32)
    for i in range(batch_number):
        if i % 50 == 0:
            print('Processing svm prediction: {:.2f}%'.format(i*100/(batch_number-1)))
        end_pos = (i+1)*batch_size
        if i == batch_number - 1:
            end_pos = f_init.shape[0]
        predict_result[i*batch_size:end_pos] = classifier.predict(f_init[i*batch_size:end_pos, :])

    # predict_result = classifier.predict(f_init)
    print("Predicting time for rf : {}s".format(time.time() - time_end))

    return predict_result.flatten().astype(np.int32)


if __name__ == '__main__':
    main()

# iris = load_iris()
#
# x = iris.data
# y = iris.target
#
# train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.8,
#                                                                   test_size=0.2)  # sklearn.model_selection.
#
# # 3.寻找最好的svm分类参数
# max_acc = 0
# parameter_ls = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
# C = 0
# gamma = 0
# for i in range(len(parameter_ls)):
#     for j in range(len(parameter_ls)):
#         classifier = svm.SVC(C=parameter_ls[i], kernel='rbf', gamma=parameter_ls[j], decision_function_shape='ovr')  # ovr:一对多策略
#         classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
#         acc = classifier.score(test_data, test_label)
#         if max_acc <= acc:
#             C = parameter_ls[i]
#             gamma = parameter_ls[j]
#             print(C, gamma, acc)
#             max_acc = acc
#
#
# classifier = svm.SVC(C=C, kernel='rbf', gamma=gamma, decision_function_shape='ovr')  # ovr:一对多策略
# classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
#
# # 4.计算svc分类器的准确率
# print("训练集：", classifier.score(train_data, train_label))
# print("测试集：", classifier.score(test_data, test_label))
#
# # 也可直接调用accuracy_score方法计算准确率
# from sklearn.metrics import accuracy_score
#
# tra_label = classifier.predict(train_data)  # 训练集的预测标签
# tes_label = classifier.predict(test_data)  # 测试集的预测标签
# print("训练集：", accuracy_score(train_label, tra_label))
# print("测试集：", accuracy_score(test_label, tes_label))
#
# # 查看决策函数
# # print('train_decision_function:\n', classifier.decision_function(train_data))  # (90,3)
# print('predict_result:\n', classifier.predict(train_data))
# print('true_result:\n', train_label)


# # -*- coding: utf-8 -*-
# # Multi-class (Nonlinear) SVM Example
# # ----------------------------------
# #
# # This function wll illustrate how to
# # implement the gaussian kernel with
# # multiple classes on the iris dataset.
# #
# # Gaussian Kernel:
# # K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
# #
# # X : (Sepal Length, Petal Width)
# # Y: (I. setosa, I. virginica, I. versicolor) (3 classes)
# #
# # Basic idea: introduce an extra dimension to do
# # one vs all classification.
# #
# # The prediction of a point will be the category with
# # the largest margin or distance to boundary.
#
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# from sklearn import datasets
# from tensorflow.python.framework import ops
#
# ops.reset_default_graph()
#
# # Load the data
# # 加载iris数据集并为每类分离目标值。
# # 因为我们想绘制结果图，所以只使用花萼长度和花瓣宽度两个特征。
# # 为了便于绘图，也会分离x值和y值
# # iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# iris = datasets.load_iris()
# x_vals = np.array([[x[0], x[3]] for x in iris.data])
# y_vals1 = np.array([1 if y == 0 else -1 for y in iris.target])
# y_vals2 = np.array([1 if y == 1 else -1 for y in iris.target])
# y_vals3 = np.array([1 if y == 2 else -1 for y in iris.target])
# y_vals = np.array([y_vals1, y_vals2, y_vals3])
# class1_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 0]
# class1_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 0]
# class2_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 1]
# class2_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 1]
# class3_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 2]
# class3_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 2]
#
# # Declare batch size
# batch_size = 50
# type_number = 3
# feature_dimension = 2
#
# # Initialize placeholders
# # 数据集的维度在变化，从单类目标分类到三类目标分类。
# # 我们将利用矩阵传播和reshape技术一次性计算所有的三类SVM。
# # 注意，由于一次性计算所有分类，
# # y_target占位符的维度是[3，None]，模型变量b初始化大小为[3，batch_size]
# x_data = tf.placeholder(shape=[None, feature_dimension], dtype=tf.float32)
# y_target = tf.placeholder(shape=[type_number, None], dtype=tf.float32)
# prediction_grid = tf.placeholder(shape=[None, feature_dimension], dtype=tf.float32)
#
# # Create variables for svm
# b = tf.Variable(tf.random_normal(shape=[3, batch_size]))
#
# # Gaussian (RBF) kernel 核函数只依赖x_data
# gamma = tf.constant(-10.0)
# dist = tf.reduce_sum(tf.square(x_data), 1)
# dist = tf.reshape(dist, [-1, 1])
# sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
# my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
#
#
# # Declare function to do reshape/batch multiplication
# # 最大的变化是批量矩阵乘法。
# # 最终的结果是三维矩阵，并且需要传播矩阵乘法。
# # 所以数据矩阵和目标矩阵需要预处理，比如xT·x操作需额外增加一个维度。
# # 这里创建一个函数来扩展矩阵维度，然后进行矩阵转置，
# # 接着调用TensorFlow的tf.batch_matmul（）函数
# def reshape_matmul(mat):
#     v1 = tf.expand_dims(mat, 1)
#     v2 = tf.reshape(v1, [type_number, batch_size, 1])
#     return tf.matmul(v2, v1)
#
#
# # Compute SVM Model 计算对偶损失函数
# first_term = tf.reduce_sum(b)
# b_vec_cross = tf.matmul(tf.transpose(b), b)
# y_target_cross = reshape_matmul(y_target)
#
# second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
# loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))
#
# # Gaussian (RBF) prediction kernel
# # 现在创建预测核函数。
# # 要当心reduce_sum（）函数，这里我们并不想聚合三个SVM预测，
# # 所以需要通过第二个参数告诉TensorFlow求和哪几个
# rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
# rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
# pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
#                       tf.transpose(rB))
# pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
#
# # 实现预测核函数后，我们创建预测函数。
# # 与二类不同的是，不再对模型输出进行sign（）运算。
# # 因为这里实现的是一对多方法，所以预测值是分类器有最大返回值的类别。
# # 使用TensorFlow的内建函数argmax（）来实现该功能
# prediction_output = tf.matmul(tf.multiply(y_target, b), pred_kernel)
# prediction = tf.arg_max(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)), tf.float32))
#
# # Declare optimizer
# my_opt = tf.train.GradientDescentOptimizer(0.01)
# train_step = my_opt.minimize(loss)
#
# # Initialize variables
# init = tf.global_variables_initializer()
#
# config = tf.ConfigProto(
#     allow_soft_placement=True)  # if the chosen device is not existed, tf can automatically distributes equipment
# config.gpu_options.allow_growth = True  # allocate memory dynamically
#
# with tf.Session(config=config) as sess:
#     sess.run(init)
#
#     # Training loop
#     loss_vec = []
#     batch_accuracy = []
#     for i in range(100):
#         rand_index = np.random.choice(len(x_vals), size=batch_size)
#         rand_x = x_vals[rand_index]
#         rand_y = y_vals[:, rand_index]
#         sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
#
#         temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
#         loss_vec.append(temp_loss)
#
#         acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
#                                                  y_target: rand_y,
#                                                  prediction_grid: rand_x})
#         batch_accuracy.append(acc_temp)
#
#         if (i + 1) % 5 == 0:
#             print('Step #' + str(i + 1))
#             print('Loss = ' + str(temp_loss))
#             print("Accu = {}".format(acc_temp))
#
#     # 创建数据点的预测网格，运行预测函数
#     x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
#     y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                          np.arange(y_min, y_max, 0.02))
#     grid_points = np.c_[xx.ravel(), yy.ravel()]
#     grid_predictions = sess.run(prediction, feed_dict={x_data: rand_x,
#                                                        y_target: rand_y,
#                                                        prediction_grid: grid_points})
#     grid_predictions = grid_predictions.reshape(xx.shape)
#
# # Plot points and grid
# plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
# plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
# plt.plot(class2_x, class2_y, 'kx', label='I. versicolor')
# plt.plot(class3_x, class3_y, 'gv', label='I. virginica')
# plt.title('Gaussian SVM Results on Iris Data')
# plt.xlabel('Pedal Length')
# plt.ylabel('Sepal Width')
# plt.legend(loc='lower right')
# plt.ylim([-0.5, 3.0])
# plt.xlim([3.5, 8.5])
# plt.show()
#
# # # Plot batch accuracy
# # plt.plot(batch_accuracy, 'k-', label='Accuracy')
# # plt.title('Batch Accuracy')
# # plt.xlabel('Generation')
# # plt.ylabel('Accuracy')
# # plt.legend(loc='lower right')
# # plt.show()
# #
# # # Plot loss over time
# # plt.plot(loss_vec, 'k-')
# # plt.title('Loss per Generation')
# # plt.xlabel('Generation')
# # plt.ylabel('Loss')
# # plt.show()
