#!/usr/bin/env python3
import tensorflow as tf

from .backend import learning_phase

class Layer():

    _count = {}

    def __init__(self,
                 activation=tf.identity,
                 kernel_initializer=tf.glorot_uniform_initializer(),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 input_shape=None):

        self.activation = activation
        self.kernel_initializer = kernel_initializer
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

    def compute_output_shape(self):
        """Return output shape."""
        raise NotImplementedError

    def __call__(self, x):
        self.build()
        return self.call(x)

class Dropout(Layer):

    def __init__(self, rate):
        self.rate = rate

        super().__init__()

    def build(self):
        pass

    def call(self, x):
        return tf.layers.dropout(x, rate=self.rate, training=learning_phase(), name=self.name)

    def compute_output_shape(self):
        return self.input_shape

class GraphConvoluation(Layer):

    def __init__(self, adj_matrix, units, *args, **kwargs):

        self.adj_matrix = adj_matrix.tocoo()
        self.units = units

        super().__init__(*args, **kwargs)

    def build(self):
        with tf.name_scope(self.name):

            indices = [[row, col] for row, col in zip(self.adj_matrix.row, self.adj_matrix.col)]

            self.sparse_adj_matrix_tensor = tf.SparseTensor(indices=indices,
                                                            values=self.adj_matrix.data,
                                                            dense_shape=self.adj_matrix.shape)

            with tf.variable_scope(self.name):
                self.kernel = tf.get_variable(
                    name='kernel',
                    initializer=self.kernel_initializer(shape=self.input_shape + (self.units,)))

                self.bias = tf.get_variable(
                    name='bias',
                    initializer=self.bias_initializer(shape=(self.units,)))

        if self.kernel_regularizer is not None:
            tf.add_to_collection('%s/regularizer' % tf.get_default_graph().get_name_scope(),
                                 tf.nn.l2_loss(self.kernel) * self.kernel_regularizer)

        if self.bias_regularizer is not None:
            tf.add_to_collection('%s/regularizer' % tf.get_default_graph().get_name_scope(),
                                 tf.nn.l2_loss(self.bias) * self.bias_regularizer)

    def call(self, x):
        with tf.name_scope(self.name):
            out = tf.sparse_tensor_dense_matmul(self.sparse_adj_matrix_tensor, x)
            out = tf.matmul(out, self.kernel)
            out = out + self.bias
            out = self.activation(out)

        return out

    def compute_output_shape(self):
        return (self.units,)

class Dense(Layer):

    def __init__(self, units, *args, **kwargs):
        self.units = units

        super().__init__(*args, **kwargs)

    def build(self):
        with tf.name_scope(self.name):
            with tf.variable_scope(self.name):
                self.kernel = tf.get_variable(
                    name='kernel',
                    initializer=self.kernel_initializer(shape=self.input_shape + (self.units,)))

                self.bias = tf.get_variable(
                    name='bias',
                    initializer=self.bias_initializer(shape=(self.units,)))

        if self.kernel_regularizer is not None:
            tf.add_to_collection('%s/regularizer' % tf.get_default_graph().get_name_scope(),
                                 tf.nn.l2_loss(self.kernel) * self.kernel_regularizer)

        if self.bias_regularizer is not None:
            tf.add_to_collection('%s/regularizer' % tf.get_default_graph().get_name_scope(),
                                 tf.nn.l2_loss(self.bias) * self.bias_regularizer)

    def call(self, x):
        with tf.name_scope(self.name):
            out = x @ self.kernel + self.bias
            out = self.activation(out)

        return out

    def compute_output_shape(self):
        return (self.units,)
