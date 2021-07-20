from typing import List
from collections.abc import Callable
from tensorflow.keras import activations, regularizers, constraints, initializers
import tensorflow as tf


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
        self.W = self.add_weight(name='weight', shape=(X_shape[1], self.units), initializers=self._krnl_initializer, trainable=True)
        self.b = self.add_weight(name='bias', shape=(self.units, ), initializers=self._bias_initiailizer, trainable=True)

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

    def train(self, idx_train, labels_train, idx_val, labels_val):
        pass

    def loss(self, train_label, true_label):
        logits = None
        _loss = tf.nn.softmax_cross_entropy_with_logits(labels=true_label, logits=logits)
        _loss = tf.reduce_mean(_loss)
        return _loss
