#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class GraphSequentialModel():

    _count = 0

    def __init__(self):
        GraphSequentialModel._count += 1

        self.name = 'Model_' + str(GraphSequentialModel._count)
        self.layers = []
        self.train_feed_dict = {}
        self.eval_feed_dict = {}

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, train_mask, validation_mask, optimizer):

        for i in range(len(self.layers)):
            if i == 0:
                continue

            self.layers[i].input_shape = self.layers[i - 1].get_output_shape()

        with tf.name_scope(self.name):
            self.x = tf.placeholder(tf.float32, shape=(None,) + self.layers[0].input_shape, name='input')
            self.y = tf.placeholder(tf.float32, shape=(None,) + self.layers[-1].get_output_shape())

            out = self.x

            for layer in self.layers:
                layer.build()
                out = layer.call(out)
                self.train_feed_dict = {**self.train_feed_dict, **layer.get_feed_dict(training=True)}
                self.eval_feed_dict = {**self.eval_feed_dict, **layer.get_feed_dict(training=False)}

            self.logits = out
            self.prediction = tf.nn.softmax(out, name='prediction')
            self.loss = loss(labels=self.y, logits=self.logits)

            if len(tf.get_collection('%s/regularizer' % tf.get_default_graph().get_name_scope())) > 0:
                self.loss += tf.add_n(tf.get_collection('%s/regularizer' % tf.get_default_graph().get_name_scope()))

            self.loss = tf.nn.embedding_lookup(self.loss, train_mask)
            self.metric = tf.nn.embedding_lookup(self.loss, validation_mask)

            self.train_step = optimizer.build(self.loss)
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

    def fit(self,x, y, epochs):

        for epoch in range(1, epochs + 1):
            feed_dict = {self.x: x, self.y: y}
            tr_loss, _ = self.session.run([self.loss, self.train_step], feed_dict={**feed_dict, **self.train_feed_dict})
            va_loss = self.session.run(self.metric, feed_dict={**feed_dict, **self.eval_feed_dict})

            if epoch % 100 == 0:
                print('Epoch: %06d, Train Loss: %.6f, Validation Loss: %.6f' % (epoch, np.mean(tr_loss), np.mean(va_loss)))

    def predict(self, x):
        return self.session.run(self.prediction, feed_dict={self.x: x})
