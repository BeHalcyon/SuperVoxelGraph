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
    def __init__(self, args):
        self.hp = args['hp']
        self.h, self.w, self.W = get_initialization(self.hp)

    def train(self, A, xs, ys):
        # 两层卷积
        h_1 = tf.nn.relu(tf.matmul(tf.matmul(A, self.h), self.w))
        h_2 = tf.matmul(A, h_1)

        #半监督部分
        xs_emb = tf.squeeze(tf.nn.embedding_lookup(h_2, xs))
        logits = tf.nn.sigmoid(tf.matmul(xs_emb, self.W))
        loss = -tf.reduce_sum(ys*tf.log(logits+ 1e-15))

        global_step = tf.train.get_or_create_global_step()

        #动态学习速率
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        # lr = self.hp.lr
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step

    def predict(self, A, xu):
        h_1 = tf.nn.relu(tf.matmul(tf.matmul(A, self.h), self.w))
        h_2 = tf.matmul(A, h_1)

        xs_emb = tf.squeeze(tf.nn.embedding_lookup(h_2, xu))
        logits = tf.nn.sigmoid(tf.matmul(xs_emb, self.W))

        pre = tf.py_func(predict_label, [logits], tf.int32)
        return pre

    def save_embeddings(self):
        flg = tf.py_func(save_emb, [self.h], tf.int32)
        return flg

