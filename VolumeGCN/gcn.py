from typing import List
from collections.abc import Callable

from numpy.lib.twodim_base import eye
from tensorflow.keras import activations, regularizers, constraints, initializers
from collections import defaultdict
import tensorflow as tf
import networkx as nx
import pickle
import scipy.sparse as sp
import numpy as np


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self,
            units: int,
            activation: Callable[[int], int] = lambda x: x,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            **kwargs):
        super(GCNLayer, self).__init__()

        self.units = units
        self.activation = activations
        self._krnl_initializer = kernel_initializer
        self._bias_initiailizer = bias_initializer

    def build(self, A_shape: tuple[int, int], X_shape: tuple[int, int]):
        """
        A_shape: shape of adjacent matrix
        X_shape: shape of feature matrix
        """
        self.W = self.add_weight(name='weight', shape=(X_shape[1], self.units), initializer=self._krnl_initializer, trainable=True)
        self.b = self.add_weight(name='bias', shape=(self.units, ), initializer=self._bias_initiailizer, trainable=True)

    def call(self, A, X):
        """
        GCN layer = AXW
        """
        self.A = X
        self.X = X
        if isinstance(self.X, tf.SparseTensor):
            h = tf.sparse.sparse_dense_matmul(self.X, self.W)
        else:
            h = tf.matmul(self.X, self.W)

        output = tf.sparse.sparse_dense_matmul(self.A, h)

        output = tf.nn.bias_add(output, self.bias)
        if callable(self.activation):
            output = self.activation(output)
        return output


class GCN():
    def __init__(self, A, X, size: List[int], learning_rate: float):
        """
        """
        self.A = A
        self.X = X
        self.layer1 = GCNLayer(size[0], activations.get('relu'))
        self.layer2 = GCNLayer(size[1])
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def train(self,labels_train, idx_train,labels_val, idx_eval):
        pass

    def loss(self, train_label, true_label):
        logits = None
        _loss = tf.nn.softmax_cross_entropy_with_logits(labels=true_label, logits=logits)
        _loss = tf.reduce_mean(_loss)
        return _loss


def load_data_planetoid(dataset):
    keys = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = defaultdict()
    for key in keys:
        with open('data_split/ind.{}.{}'.format(dataset, key), 'rb') as f:
            objects[key] = pickle.load(f, encoding='latin1')
    test_index = [int(x) for x in open('data_split/ind.{}.test.index'.format(dataset))]
    test_index_sort = np.sort(test_index)
    G = nx.from_dict_of_lists(objects['graph'])

    A_mat = nx.adjacency_matrix(G)
    X_mat = sp.vstack((objects['allx'], objects['tx'])).tolil()
    X_mat[test_index, :] = X_mat[test_index_sort, :]
    z_vec = np.vstack((objects['ally'], objects['ty']))
    z_vec[test_index, :] = z_vec[test_index_sort, :]
    z_vec = z_vec.argmax(1)

    train_idx = range(len(objects['y']))
    val_idx = range(len(objects['y']), len(objects['y']) + 500)
    test_idx = test_index_sort.tolist()

    return A_mat, X_mat, z_vec, train_idx, val_idx, test_idx

def preprocess_graph(A):
    A_bar = A + sp.eye(A.shape[0])  # A' = A + I
    D = A_bar.sum(1).A1
    D_half = sp.diags(np.power(D, -0.5))
    return (D_half @ A_bar @ D_half).tocsr()

if __name__ == '__main__':
    A_mat, X_mat, z_vec, train_idx, val_idx, test_idx = load_data_planetoid("cora")
    A = preprocess_graph(A_mat)
    print(A.shape, X_mat.shape, len(z_vec))
    test_layer = GCNLayer(16)
    test_layer.build(A_mat.shape, X_mat.shape)