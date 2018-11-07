#!/usr/bin/env python3
import sys
import os

import numpy as np
from sklearn.metrics import accuracy_score

from utils.tfpkg.models import Evaluator
from utils.io import read_pkl

model_path = sys.argv[1]

nodes = read_pkl('tmp/nodes.pkl')
loc_db = read_pkl('tmp/location.pkl')
categorical = read_pkl('tmp/categorical.pkl')
candidate = read_pkl('tmp/candidate.pkl')
train_mask = read_pkl('%s/train.mask.pkl' % model_path)
validation_mask = read_pkl('%s/validation.mask.pkl' % model_path)
node_features = read_pkl('tmp/features.pkl')
node_labels = read_pkl('tmp/labels.pkl')
user_checkins = read_pkl('tmp/user_checkins.pkl')

def top_k_accuracy(y_true, y_pred, k):
    total = y_true.shape[0]
    p = 0

    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
    ground_truth = np.argmax(y_true, axis=1)

    for label, candidates in zip(ground_truth, top_k_indices):
        if label in candidates:
            p += 1

    return p / total

def get_test_mask():
    u_m_pair = read_pkl('tmp/user_miss_pair.pkl')
    nodes = read_pkl('tmp/nodes.pkl')

    return [nodes.index(el) for el in u_m_pair]

def find_place(places, tag, node_idx):

    ret = []

    for place in places:
        if loc_db[place]['class'] == tag:
            ret.append(place)

    if len(ret) == 0:
        return ret

    # sort by group
    group_order = np.argsort(node_features[node_idx][-6:]).tolist()
    ret.sort(key=lambda x: group_order.index(loc_db[x]['group']))
    ret.reverse()

    return ret

def decrease_visited(place_list, user):
    length = len(place_list)
    i = 0

    while i < length:
        place = place_list[i]

        if place not in user_checkins[user]:
            i += 1
            continue

        place_list.remove(place)
        place_list.append(place)
        length -= 1

    return place_list

if __name__ == '__main__':

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

    if not os.path.isdir('result'):
        os.makedirs('result')

    test_mask = get_test_mask()
    f = open('result/ans.txt', 'w')

    for i in test_mask:
        user = nodes[i][:-2]
        pred = proba[i]
        sort_idx = np.argsort(pred)

        place_list = []

        for j in range(len(sort_idx) - 1, -1, -1):
            c = categorical[sort_idx[j]]
            places = find_place(places=candidate, tag=c, node_idx=i)
            place_list.extend(places)

        place_list = decrease_visited(place_list, user)
        f.write('%s:%s\n' % (user, ','.join(place_list)))

    f.close()
