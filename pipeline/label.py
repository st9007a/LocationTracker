#!/usr/bin/env python3
from pprint import pprint

import numpy as np
from scipy import sparse

from utils.io import read_pkl, save_pkl

loc_db = read_pkl('tmp/location.pkl')
candidate = read_pkl('tmp/candidate.pkl')
nodes = read_pkl('tmp/nodes.pkl')
tag2class = read_pkl('tmp/tag2class.pkl')

def get_train_class_weight():
    counter = {}

    for node in nodes:
        if node[-1] == '?':
            continue

        c = loc_db[node]['tag']

        if c not in counter:
            counter[c] = 0

        counter[c] += 1

    return counter

    # counter_list = [(el, counter[el]) for el in counter]
    # counter_list.sort(key=lambda x:x[1])
    # counter_list.reverse()
    #
    # return counter_list

def get_test_class_weight():
    counter = {}

    for node in candidate:

        c = loc_db[node]['tag']

        if c not in counter:
            counter[c] = 0

        counter[c] += 1
    return counter

    # counter_list = [(el, counter[el]) for el in counter]
    # counter_list.sort(key=lambda x:x[1])
    # counter_list.reverse()
    #
    # return counter_list

if __name__ == '__main__':

    tr_class_weight = get_train_class_weight()
    te_class_weight = get_test_class_weight()
    # pprint(te_class_weight)
    # exit()

    keep_tags = set()
    keep_classes = {}
    grow_classes = set()

    for tr_tag in tr_class_weight:
        if tr_tag in te_class_weight or tr_class_weight[tr_tag] > 80:
            if 'restaurant' in tr_tag and (tr_tag not in te_class_weight):
                continue
            keep_tags.add((tr_tag, tr_class_weight[tr_tag]))

    for node in nodes:
        if node[-1] == '?':
            continue

        t = loc_db[node]['tag']

        if t not in [el[0] for el in keep_tags]:
            continue

        c = loc_db[node]['class']

        if c not in keep_classes:
            keep_classes[c] = 0
        keep_classes[c] += 1

    categorical = list(keep_classes)
    # keep_classes = [(el, keep_classes[el]) for el in keep_classes]
    # keep_classes.sort(key=lambda x:x[1])
    # keep_classes.reverse()
    # print(len(keep_classes))

    train_mask = []
    labels = np.zeros((len(nodes), len(categorical)))
    for i, node in enumerate(nodes):
        if node[-1] == '?':
            continue

        c = loc_db[node]['class']
        if c in categorical:
            train_mask.append(i)
            labels[i][categorical.index(c)] = 1

    save_pkl('tmp/categorical.pkl', categorical)
    save_pkl('tmp/train_mask.pkl', train_mask)
    save_pkl('tmp/labels.pkl', labels)
