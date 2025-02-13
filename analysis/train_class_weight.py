#!/usr/bin/env python3
from pprint import pprint

import numpy as np

from utils.io import read_pkl

loc_db = read_pkl('tmp/location.pkl')
categorical = read_pkl('tmp/categorical.pkl')
labels = read_pkl('tmp/labels.pkl')
nodes = read_pkl('tmp/nodes.pkl')
u_m_pair = read_pkl('tmp/user_miss_pair.pkl')

def get_test_mask():
    return [nodes.index(el) for el in u_m_pair]

if __name__ == '__main__':

    labels = np.argmax(labels, axis=1)
    test_mask = get_test_mask()

    counter = {}

    for place in nodes:
        if place[-1] == '?':
            continue

        c = loc_db[place]['tag']

        if c not in counter:
            counter[c] = 0

        counter[c] += 1

    counter_list = [(k, counter[k]) for k in counter]
    counter_list.sort(key=lambda x:x[1])
    counter_list.reverse()
    # counter_list = list(filter(lambda x: x[1] > 80, counter_list))
    counter_list = [(el[0], el[1] / labels.shape[0]) for el in counter_list]
    pprint(counter_list)
