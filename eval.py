#!/usr/bin/env python3
import numpy as np

from utils.tfpkg.models import Evaluator
from utils.io import read_pkl

if __name__ == '__main__':

    test_mask = read_pkl('features/test.mask.pkl')
    node_features = np.load('features/nodes.npy')
    node_labels = np.load('features/labels.npy')

    model = Evaluator('models/test')
    model.eval(node_features)
