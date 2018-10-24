#!/usr/bin/env python3
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

    train_mask = read_pkl('features/train.mask.pkl')
    validation_mask = read_pkl('features/validation.mask.pkl')
    test_mask = read_pkl('features/test.mask.pkl')
    node_features = np.load('features/nodes.npy')
    node_labels = np.load('features/labels.npy')

    model = Evaluator('models/test')
    proba = model.eval(node_features)

    print(top_k_accuracy(node_labels[train_mask], proba[train_mask], k=1))
    print(top_k_accuracy(node_labels[validation_mask], proba[validation_mask], k=1))

    print()

    print(top_k_accuracy(node_labels[train_mask], proba[train_mask], k=3))
    print(top_k_accuracy(node_labels[validation_mask], proba[validation_mask], k=3))

    print()

    print(top_k_accuracy(node_labels[train_mask], proba[train_mask], k=5))
    print(top_k_accuracy(node_labels[validation_mask], proba[validation_mask], k=5))

    print()

    print(top_k_accuracy(node_labels[train_mask], proba[train_mask], k=30))
    print(top_k_accuracy(node_labels[validation_mask], proba[validation_mask], k=30))
