#!/usr/bin/env python3
import sys
import os

import numpy as np
from sklearn.metrics import accuracy_score

from utils.tfpkg.models import Evaluator
from utils.io import read_pkl

model_path = sys.argv[1]

with open('tmp/numclasses.txt', 'r') as f:
    num_classes = int(f.read().rstrip('\n'))

nodes = read_pkl('tmp/nodes.pkl')
loc_db = read_pkl('tmp/location.pkl')
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

    models = [Evaluator(model_path + '/' + str(i)) for i in range(5)]
    proba = sum([model.eval(node_features) for model in models]) / 5

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
            places = find_place(places=candidate, tag=sort_idx[j], node_idx=i)
            place_list.extend(places)

        place_list = decrease_visited(place_list, user)
        f.write('%s:%s\n' % (user, ','.join(place_list)))

    f.close()
