#!/usr/bin/env python3
from pprint import pprint

from utils.io import read_pkl

candidate = read_pkl('tmp/candidate.pkl')

if __name__ == '__main__':

    counter = {}

    for lid in candidate:
        c = candidate[lid]['class']

        if c not in counter:
            counter[c] = 0

        counter[c] += 1

    counter_list = [(k, counter[k]) for k in counter]
    counter_list.sort(key=lambda x:x[1])
    counter_list.reverse()
    counter_list = [(el[0], el[1] / 100) for el in counter_list]
    pprint(counter_list)
