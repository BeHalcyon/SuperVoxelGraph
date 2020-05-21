import tensorflow as tf
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

def emb_value_cos(x, y):
    a_b = np.dot(x, y)
    a = np.fabs(sum(x ** 2) ** 0.5)
    b = np.fabs(sum(y ** 2) ** 0.5)
    # print(str(x)+"    "+str(y))
    a_dev_b = a_b / (a * b)

    return a_dev_b.reshape(1)

def emb_value_weight_2(x, y):
    a_b = np.dot(x, y)

    return a_b.reshape(1)

def node_classification(x, x_label, y, y_lable):
    cls = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=np.random.RandomState(0)))
    cls.fit(x, x_label)

def get_initialization(hp):
    h_init = (np.random.randn(hp.node_num, hp.dim) / np.sqrt(hp.node_num/2)).astype('float32')
    w_init = (np.random.randn(hp.dim, hp.dim) / np.sqrt(hp.dim/2)).astype('float32')
    W_init = (np.random.randn(hp.dim, hp.label) / np.sqrt(hp.dim/2)).astype('float32')
    # y = []
    # for t in range(hp.T):
    #     y.append(tf.Variable(h_init + 0.001 * tf.random_normal([hp.node_num, hp.dim]) / np.sqrt(hp.node_num/2),
    #                          name='emb_'+str(t), trainable=True))
    y = tf.Variable(h_init, name='emb', trainable=True)
    w = tf.Variable(w_init, name='w', trainable=True)
    W = tf.Variable(W_init, name='W', trainable=True)
    return y, w, W

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def predict_label(y):
    y[y>=0.5]=1.0
    y[y<0.5]=0.0
    return y

def Graphs(hp):
    G = nx.read_gexf(hp.dataset)
    return G

