#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Layer():

    _count = {}

    def __init__(self,
                 activation=tf.identity,
                 kernel_initializer=tf.glorot_uniform_initializer,
                 bias_initializer=tf.zeros_initializer,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 input_shape=None):

        self.activation = activation,
        self.kernel_initilaizer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.input_shape = input_shape

        if self.__class__.__name__ not in Layer._count:
            Layer._count[self.__class__.__name__] = 0
        Layer._count[self.__class__.__name__] += 1

        self.name = '%s_%d' % (self.__class__.__name__, Layer._count[self.__class__.__name__])

    def build(self):
        """Initialization and preprocessing."""
        raise NotImplementedError

    def call(self, x):
        """Build tensorflow operation."""
        raise NotImplementedError

    def get_output_shape(self):
        """Return output shape."""
        raise NotImplementedError

    def get_feed_dict(self, training):
        """Return feed dict"""
        raise NotImplementedError

class Dropout(Layer):

    def __init__(self, rate):
        self.rate = rate

        super().__init__()

    def build(self):
        with tf.name_scope(self.name):
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def call(self, x):
        with tf.name_scope(self.name):
            out = tf.nn.dropout(x, self.keep_prob)

        return out

    def get_output_shape(self):
        return self.input_shape

    def get_feed_dict(self, training):
        if training:
            return {self.keep_prob: 1 - self.rate}
        else:
            return {self.keep_prob: 1}

class GraphConvoiuation(Layer):

    def __init__(self, adj_matrix, units, *args, **kwargs):

        self.adj_matrix = adj_matrix
        self.units = units

        super().__init__(*args, **kwargs)
