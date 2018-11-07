#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf

from .backend import learning_phase

LEARNING_PHASE = 'learning_phase'
SIGNATURE_INPUT = 'input'
SIGNATURE_OUTPUT = 'output'
SIGNATURE_METHOD_NAME = 'prediction'
SIGNATURE_KEY = 'prediction'

def top_k_accuracy(y_true, y_pred, k):
    total = y_true.shape[0]
    p = 0

    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
    ground_truth = np.argmax(y_true, axis=1)

    for label, candidates in zip(ground_truth, top_k_indices):
        if label in candidates:
            p += 1

    return p / total

class GraphSequentialModel():

    _count = 0

    def __init__(self):
        GraphSequentialModel._count += 1

        self.name = 'Model_' + str(GraphSequentialModel._count)
        self.layers = []
        self.saver = None
        self.builder = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, train_mask, validation_mask, optimizer):

        for i in range(1, len(self.layers)):
            self.layers[i].input_shape = self.layers[i - 1].compute_output_shape()

        with tf.name_scope(self.name):
            self.x = tf.placeholder(tf.float32, shape=(None,) + self.layers[0].input_shape, name='input')
            self.y = tf.placeholder(tf.float32, shape=(None,) + self.layers[-1].compute_output_shape())

            out = self.x

            for layer in self.layers:
                out = layer(out)

            self.logits = out
            self.prediction = tf.nn.softmax(out, name='prediction')
            self.loss = loss(labels=self.y, logits=self.logits)

            if len(tf.get_collection('%s/regularizer' % tf.get_default_graph().get_name_scope())) > 0:
                self.loss += tf.add_n(tf.get_collection('%s/regularizer' % tf.get_default_graph().get_name_scope()))

            self.loss = tf.nn.embedding_lookup(self.loss, train_mask)
            self.metric = tf.nn.embedding_lookup(self.loss, validation_mask)

            self.train_mask = train_mask
            self.validation_mask = validation_mask

            self.train_step = optimizer.build(self.loss)
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

    def fit(self,x, y, epochs, save_path, k):
        max_top_k_acc = 0
        checkpoint_dir = save_path + '/checkpoint'
        checkpoint_path = checkpoint_dir + '/model.ckpt'
        savedmodel_path = save_path + '/build'

        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for epoch in range(1, epochs + 1):
            self.session.run(self.train_step, feed_dict= {self.x: x, self.y: y, learning_phase(): True})

            if epoch % 100 == 0:
                prob = self.session.run(self.prediction, feed_dict= {self.x: x, self.y: y, learning_phase(): False})
                top_k_acc = top_k_accuracy(y[self.validation_mask], prob[self.validation_mask], k=k)

                if max_top_k_acc < top_k_acc:
                    max_top_k_acc = top_k_acc
                    self.save_checkpoint(checkpoint_path)

                # print('Epoch: %06d, Train Loss: %.6f, Validation Loss: %.6f, Best Va Loss: %.6f' % (epoch, tr_loss, va_loss, min_va_loss))
                print('Epoch: %06d, Validation Acc: %.6f, Best Acc: %.6f' % (epoch, top_k_acc, max_top_k_acc))

        self.load_checkpoint(checkpoint_path)
        self.serve(savedmodel_path)

    def save_checkpoint(self, path):
        if self.saver is None:
            self.saver = tf.train.Saver()

        self.saver.save(self.session, path)

    def load_checkpoint(self, path):
        if self.saver is None:
            self.saver = tf.train.Saver()
        self.saver.restore(self.session, path)

    def serve(self, path):
        if self.builder is None:
            self.builder = tf.saved_model.builder.SavedModelBuilder(path)

        inputs = {LEARNING_PHASE: tf.saved_model.utils.build_tensor_info(learning_phase()),
                  SIGNATURE_INPUT: tf.saved_model.utils.build_tensor_info(self.x)}

        outputs = {SIGNATURE_OUTPUT: tf.saved_model.utils.build_tensor_info(self.prediction)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, SIGNATURE_METHOD_NAME)
        self.builder.add_meta_graph_and_variables(self.session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={SIGNATURE_KEY: signature})
        self.builder.save()

class Evaluator():

    def __init__(self, path):
        tf.reset_default_graph()

        self.session = tf.Session()

        meta_graph_def = tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], '%s/build/' % path)
        signature = meta_graph_def.signature_def

        input_tensor_name = signature[SIGNATURE_KEY].inputs[SIGNATURE_INPUT].name
        learning_phase_tensor_name = signature[SIGNATURE_KEY].inputs[LEARNING_PHASE].name
        output_tensor_name = signature[SIGNATURE_KEY].outputs[SIGNATURE_OUTPUT].name

        self.input_holder = self.session.graph.get_tensor_by_name(input_tensor_name)
        self.prediction = self.session.graph.get_tensor_by_name(output_tensor_name)
        self.learning_phase = self.session.graph.get_tensor_by_name(learning_phase_tensor_name)

    def eval(self, x_eval):
        return self.session.run(self.prediction, feed_dict={self.input_holder: x_eval, self.learning_phase: False})
