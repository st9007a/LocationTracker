#!/usr/bin/env python3
import sys
import os

import numpy as np
from sklearn.metrics import accuracy_score

from utils.tfpkg.models import Evaluator
from utils.io import read_pkl
from utils.location import distance

model_path = sys.argv[1]

with open('tmp/numclasses.txt', 'r') as f:
    num_classes = int(f.read().rstrip('\n'))

nodes = read_pkl('tmp/nodes.pkl')
loc_db = read_pkl('tmp/location.pkl')
candidate = read_pkl('tmp/candidate.pkl')
node_features = read_pkl('tmp/features.pkl')
node_labels = read_pkl('tmp/labels.pkl')
user_checkins = read_pkl('tmp/user_checkins.pkl')
user_miss_loc = read_pkl('tmp/user_miss_loc.pkl')
categorical = read_pkl('tmp/categorical.pkl')

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

def get_average_distance(user, place):
    if len(user_miss_loc[user]) == 0:
        return 0

    tmp = [distance(el[0], el[1], loc_db[place]['lat'], loc_db[place]['lon']) for el in user_miss_loc[user]]
    return sum(tmp) / len(tmp)

def find_place(places, tag, node_idx):

    user = nodes[node_idx][:-2]
    user_relative_loc = user_miss_loc[user]
    tmp = []

    for place in places:
        if loc_db[place]['class'] == tag:
            tmp.append(place)

    if len(tmp) == 0:
        return tmp

    # sort by group
    group_order = np.argsort(node_features[node_idx][-6:]).tolist()[::-1]
    ret = []

    for g in group_order:
        #sort by lat, lon
        match = [el for el in tmp if loc_db[el]['group'] == g]
        match.sort(key=lambda x: get_average_distance(user, x))

        ret.extend(match)

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
    f1 = open('result/ans.txt', 'w')
    f2 = open('result/check.txt', 'w')

    for i in test_mask:
        user = nodes[i][:-2]
        pred = proba[i]
        sort_idx = np.argsort(pred)[::-1]

        place_list = []
        tags = []

        for j in range(len(sort_idx)):
            places = find_place(places=candidate, tag=categorical[sort_idx[j]], node_idx=i)
            place_list.extend(places)

            if len(places) != 0:
                tags.append((categorical[sort_idx[j]], len(places)))

        place_list = decrease_visited(place_list, user)
        f1.write('%s:%s\n' % (user, ','.join(place_list)))
        f2.write(str(tags) + '\n')
        print(len(tags))

    f1.close()
    f2.close()
