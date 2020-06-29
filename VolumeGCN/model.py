import tensorflow as tf
import numpy as np
import os
import sys
from args import parse_args
arg = parse_args()
from module import get_initialization, noam_scheme, predict_label

def save_emb(y, alpha):
    file_path = '../Embeddings/Volume-GCN/' + arg.dataset + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    for t in range(arg.T):
        file_nam = file_path+str(t).zfill(4) + '.emb'
        file = open(file_nam, 'w')
        for v in range(arg.node_num):
            file.write(str(v))
            for it in y[t][v]:
                file.write(' ' + str(it))
            file.write('\n')
        file.close()
    file_nam = file_path+'alpha.con'
    file = open(file_nam, 'w')
    file.write(str(arg.node_num)+' '+str(arg.dim)+'\n')
    for v in range(arg.node_num):
        file.write(str(v))
        for it in alpha[v]:
            file.write(' ' + str(it))
        file.write('\n')
    file.close()
    return 1

class Volume_GCN:
    def __init__(self, args, G):
        self.hp = args['hp']
        self.G = G
        # self.f, self.w1, self.w2, self.W = get_initialization(self.hp, self.G)
        self.f = get_initialization(self.hp, self.G)

    # 设定参数
    def train(self, A, xs, ys):
        # # 先映射
        # self.h = tf.layers.dense(self.f, self.hp.dim, activation=None)

        # 两层卷积
        h_1 = tf.matmul(A, self.f)
        h_1 = tf.layers.dense(h_1, self.hp.hidden1, activation=tf.nn.relu, name="hidden_layer_1", reuse=tf.AUTO_REUSE)
        h_1 = tf.layers.dropout(h_1, self.hp.dropout, name="dropout_layer_1")

        h_2 = tf.matmul(A, h_1)
        h_2 = tf.layers.dense(h_2, self.hp.hidden2, activation=tf.nn.relu, name="hidden_layer_2", reuse=tf.AUTO_REUSE)
        h_2 = tf.layers.dropout(h_2, self.hp.dropout, name="dropout_layer_2")

        #半监督部分
        xs_emb = tf.squeeze(tf.nn.embedding_lookup(h_2, xs))
        logits = tf.layers.dense(xs_emb, self.hp.label, activation=None, name="classifer", reuse=tf.AUTO_REUSE)
        labels = tf.one_hot(ys, self.hp.label, axis=1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)

        # 这个函数主要用于返回或者创建（如果有必要的话）一个全局步数的tensor。参数只有一个，就是图，如果没有指定那么就是默认的图。
        global_step = tf.train.get_or_create_global_step()

        #动态学习速率
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        # lr = self.hp.lr
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step

    def predict(self, A, xu):
        h_1 = tf.matmul(A, self.f)
        h_1 = tf.layers.dense(h_1, self.hp.hidden1, activation=tf.nn.relu, name="hidden_layer_1", reuse=tf.AUTO_REUSE)

        h_2 = tf.matmul(A, h_1)
        h_2 = tf.layers.dense(h_2, self.hp.hidden2, activation=tf.nn.relu, name="hidden_layer_2", reuse=tf.AUTO_REUSE)

        xs_emb = tf.squeeze(tf.nn.embedding_lookup(h_2, xu))
        logits = tf.layers.dense(xs_emb, self.hp.label, activation=None, name="classifer", reuse=tf.AUTO_REUSE)

        pre = tf.argmax(logits, 1)
        return pre

    # binary classification prediction
    def binaryPredict(self, A, xu):
        h_1 = tf.matmul(A, self.f)
        h_1 = tf.layers.dense(h_1, self.hp.hidden1, activation=tf.nn.relu, name="hidden_layer_1", reuse=tf.AUTO_REUSE)

        h_2 = tf.matmul(A, h_1)
        h_2 = tf.layers.dense(h_2, self.hp.hidden2, activation=tf.nn.relu, name="hidden_layer_2", reuse=tf.AUTO_REUSE)

        xs_emb = tf.squeeze(tf.nn.embedding_lookup(h_2, xu))
        logits = tf.layers.dense(xs_emb, self.hp.label, activation=tf.nn.sigmoid, name="classifer", reuse=tf.AUTO_REUSE)

        # pre = tf.argmax(logits, 1)
        return logits

    def save_embeddings(self):
        flg = tf.py_func(save_emb, [self.h], tf.int32)
        return flg