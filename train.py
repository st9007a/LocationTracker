#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import train_test_split

from utils.tfpkg.layers import Dropout, GraphConvoluation
from utils.tfpkg.models import GraphSequentialModel
from utils.tfpkg.optimizers import Optimizer
from utils.io import read_pkl

def get_mask(total_size, validation_ratio, exclude_idx):
    mask = [i for i in range(total_size) if i not in exclude_idx]

    return train_test_split(mask, test_size=validation_ratio)

if __name__ == '__main__':

    test_mask = read_pkl('features/test.mask.pkl')
    node_features = np.load('features/nodes.npy')
    node_labels = np.load('features/labels.npy')
    adj_matrix = sparse.load_npz('features/graph.npz')

    train_mask, validation_mask = get_mask(total_size=node_features.shape[0], validation_ratio=0.2, exclude_idx=test_mask)

    model = GraphSequentialModel()
    model.add(GraphConvoluation(adj_matrix=adj_matrix, units=128, activation=tf.nn.relu, input_shape=(24,)))
    model.add(GraphConvoluation(adj_matrix=adj_matrix, units=128, activation=tf.nn.relu))
    model.add(GraphConvoluation(adj_matrix=adj_matrix, units=128, activation=tf.nn.relu))
    model.add(GraphConvoluation(adj_matrix=adj_matrix, units=node_labels.shape[1]))

    optimizer = Optimizer(lr=1e-2, tf_optimizer=tf.train.RMSPropOptimizer, decay=0.9999)

    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits_v2,
                  train_mask=train_mask, validation_mask=validation_mask, optimizer=optimizer)
    model.fit(node_features, node_labels, epochs=350000)
