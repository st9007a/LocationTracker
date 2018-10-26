#!/usr/bin/env python3
import os
import sys

import numpy as np
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import train_test_split

from utils.tfpkg.layers import Dropout, GraphConvoluation, Dense
from utils.tfpkg.models import GraphSequentialModel
from utils.tfpkg.optimizers import Optimizer
from utils.io import read_pkl, save_pkl

def get_mask(total_size, validation_ratio, exclude_idx):
    mask = [i for i in range(total_size) if i not in exclude_idx]

    return train_test_split(mask, test_size=validation_ratio)

def get_test_mask():
    u_m_pair = read_pkl('tmp/user_miss_pair.pkl')
    nodes = read_pkl('tmp/nodes.pkl')

    return [nodes.index(el) for el in u_m_pair]

if __name__ == '__main__':
    model_path = sys.argv[1]

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    test_mask = get_test_mask()
    node_features = read_pkl('tmp/features.pkl')
    node_labels = read_pkl('tmp/labels.pkl')
    adj_matrix = sparse.load_npz('tmp/graph.npz')

    train_mask, validation_mask = get_mask(total_size=node_features.shape[0], validation_ratio=0.2, exclude_idx=test_mask)
    save_pkl('%s/train.mask.pkl' % model_path, train_mask)
    save_pkl('%s/validation.mask.pkl' % model_path, validation_mask)

    model = GraphSequentialModel()
    model.add(GraphConvoluation(adj_matrix=adj_matrix, units=256, activation=tf.nn.relu, kernel_regularizer=5e-5, bias_regularizer=5e-5, input_shape=(30,)))
    model.add(GraphConvoluation(adj_matrix=adj_matrix, units=512, activation=tf.nn.relu, kernel_regularizer=5e-5, bias_regularizer=5e-5))
    model.add(Dropout(0.8))
    model.add(GraphConvoluation(adj_matrix=adj_matrix, units=node_labels.shape[1]))

    optimizer = Optimizer(lr=1e-2, tf_optimizer=tf.train.RMSPropOptimizer, decay=0.997)

    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits_v2,
                  train_mask=train_mask, validation_mask=validation_mask, optimizer=optimizer)
    # model.load_checkpoint('models/test3/checkpoint/model.ckpt')
    # model.serve('models/test3/build')
    model.fit(node_features, node_labels, epochs=5000, save_path=model_path, k=1)
