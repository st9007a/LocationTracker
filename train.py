#!/usr/bin/env python3
import os
import sys
from random import shuffle

import numpy as np
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import train_test_split

from utils.tfpkg.layers import Dropout, GraphConvolution, Dense
from utils.tfpkg.models import GraphSequentialModel
from utils.tfpkg.optimizers import Optimizer
from utils.io import read_pkl, save_pkl

def get_mask(total_size, validation_ratio, exclude_idx):
    mask = [i for i in range(total_size) if i not in exclude_idx]

    return train_test_split(mask, test_size=validation_ratio)

def get_k_fold_mask(total_size, folds, exclude_idx):
    mask = [i for i in range(total_size) if i not in exclude_idx]
    shuffle(mask)

    ret = []
    mod = (total_size - len(exclude_idx)) % folds
    base = (total_size - len(exclude_idx) - mod) // folds

    for i in range(folds):
        ret.append([])
        ret[i].extend(mask[base * i:base * (i+1)])

    for i, m in enumerate(range(mod)):
        ret[i].append(mask[base * folds + m])

    return ret

def get_test_mask():
    u_m_pair = read_pkl('tmp/user_miss_pair.pkl')
    nodes = read_pkl('tmp/nodes.pkl')

    return [nodes.index(el) for el in u_m_pair]

if __name__ == '__main__':
    root_path = sys.argv[1]

    if not os.path.isdir(root_path):
        os.makedirs(root_path)

    test_mask = get_test_mask()
    node_features = read_pkl('tmp/features.pkl')
    node_labels = read_pkl('tmp/labels.pkl')
    adj_matrix = sparse.load_npz('tmp/graph.npz')

    masks = get_k_fold_mask(total_size=node_features.shape[0], folds=5, exclude_idx=test_mask)

    perf = []

    for i in range(5):
        model_path = root_path + '/' + str(i)
        os.makedirs(model_path)

        train_mask = [masks[j] for j in range(5) if j != i]
        train_mask = np.array(train_mask, dtype=int)
        train_mask = np.reshape(train_mask, [-1])

        validation_mask = np.array(masks[i], dtype=int)

        model = GraphSequentialModel(adj_matrix=adj_matrix)
        model.add(GraphConvolution(units=256, activation=tf.nn.relu, kernel_regularizer=2e-4, bias_regularizer=2e-4, input_shape=(30,)))
        model.add(GraphConvolution(units=512, activation=tf.nn.relu, kernel_regularizer=2e-4, bias_regularizer=2e-4))
        model.add(Dropout(0.8))
        model.add(GraphConvolution(units=node_labels.shape[1]))

        optimizer = Optimizer(lr=1e-2, tf_optimizer=tf.train.RMSPropOptimizer, decay=0.997)

        model.compile(loss=tf.nn.softmax_cross_entropy_with_logits_v2,
                      train_mask=train_mask, validation_mask=validation_mask, optimizer=optimizer)
        val_acc = model.fit(node_features, node_labels, epochs=5000, save_path=model_path, k=1)
        perf.append(val_acc)

    with open('%s/perf.txt' % root_path, 'w') as f:
        f.write(str(sum(perf) / 5) + '\n')

    print(perf)
    print(sum(perf) / 5)
