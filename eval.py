#!/usr/bin/env python3
import sys

import numpy as np
from sklearn.metrics import accuracy_score

from utils.tfpkg.models import Evaluator
from utils.io import read_pkl

def top_k_accuracy(y_true, y_pred, k):
    total = y_true.shape[0]
    p = 0

    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
    ground_truth = np.argmax(y_true, axis=1)

    for label, candidates in zip(ground_truth, top_k_indices):
        if label in candidates:
            p += 1

    return p / total

if __name__ == '__main__':
    model_path = sys.argv[1]

    train_mask = read_pkl('%s/train.mask.pkl' % model_path)
    validation_mask = read_pkl('%s/validation.mask.pkl' % model_path)
    test_mask = read_pkl('features/test.mask.pkl')
    node_features = np.load('features/nodes.npy')
    node_labels = np.load('features/labels.npy')

    model = Evaluator(model_path)
    proba = model.eval(node_features)

    print('top 1 train acc:', top_k_accuracy(node_labels[train_mask], proba[train_mask], k=1))
    print('top 1 validation acc:', top_k_accuracy(node_labels[validation_mask], proba[validation_mask], k=1))

    print()

    print('top 3 train acc:', top_k_accuracy(node_labels[train_mask], proba[train_mask], k=3))
    print('top 3 validation acc:', top_k_accuracy(node_labels[validation_mask], proba[validation_mask], k=3))

    print()

    print('top 5 train acc:', top_k_accuracy(node_labels[train_mask], proba[train_mask], k=5))
    print('top 5 validation acc:', top_k_accuracy(node_labels[validation_mask], proba[validation_mask], k=5))

    print()

    print('top 30 train acc:', top_k_accuracy(node_labels[train_mask], proba[train_mask], k=30))
    print('top 30 validation acc:', top_k_accuracy(node_labels[validation_mask], proba[validation_mask], k=30))
