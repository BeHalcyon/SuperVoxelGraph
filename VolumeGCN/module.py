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

# def node_classification(x, x_label, y, y_lable):
#     cls = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=np.random.RandomState(0)))
#     cls.fit(x, x_label)


def node_classification(x, x_label, y):
    cls = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=np.random.RandomState(0)))
    cls.fit(x, x_label)
    return cls.predict(y)


def get_initialization(hp, G):
    # h_init = (np.random.randn(hp.node_num, hp.dim) / np.sqrt(hp.node_num/2)).astype('float32')
    f_init = np.zeros((hp.node_num, hp.vec_dim), dtype=np.float32)

    print('The dimension of each node : {}'.format(hp.vec_dim))

    for n in range(hp.node_num):
        for i in range(hp.vec_dim):
            # print(n, i)
            f_init[n][i] = G.node[str(n)][str(i)]

    # w1_init = (np.random.randn(hp.dim, hp.hidden1) / np.sqrt(hp.dim/2)).astype('float32')
    # w2_init = (np.random.randn(hp.hidden1, hp.dim) / np.sqrt(hp.dim / 2)).astype('float32')
    # W_init = (np.random.randn(hp.dim, hp.label) / np.sqrt(hp.dim/2)).astype('float32')
    # y = []
    # for t in range(hp.T):
    #     y.append(tf.Variable(h_init + 0.001 * tf.random_normal([hp.node_num, hp.dim]) / np.sqrt(hp.node_num/2),
    #                          name='emb_'+str(t), trainable=True))
    f = tf.Variable(f_init, name='emb', trainable=True)
    # w1 = tf.Variable(w1_init, name='w1', trainable=True)
    # w2 = tf.Variable(w2_init, name='w2', trainable=True)
    # W = tf.Variable(W_init, name='W', trainable=True)
    return f
    # return f, w1, w2, W

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def predict_label(y):
    y = y.reshape([-1])
    y[y>=0.5]=1.0
    y[y<0.5]=0.0

    return y

def Graphs(hp):
    G = nx.read_gexf(hp.dataset)
    return G


from sklearn.metrics import f1_score, precision_score, recall_score
def metricMeasure(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='macro')
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    return p, r, f1
