#!/usr/bin/env python3
import os

import numpy as np

from utils.io import save_pkl

chars = 'abcdefghijklmnopqrstuvwxyz'

def word_normalize(word):
    return ''.join([c for c in word.lower() if c in chars])

if __name__ == '__main__':
    in_file = 'raw/loc_id_info.txt'
    out_dir = 'tmp'

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    loc_db = {}
    # tag2class = {}
    # ptr = -1
    #
    # with open(in_file_1, 'r') as f:
    #     for line in f:
    #         if line[0] != ' ':
    #             ptr += 1
    #             continue
    #
    #         tag2class[word_normalize(line)] = ptr
    # print(tag2class)

    with open(in_file, 'r') as f:

        for line in f:
            data = line.rstrip('\n').split('\t')
            loc_db[data[0]] = {
                'lat': float(data[1]),
                'lon': float(data[2]),
                'tag': word_normalize(data[3]),
                'country': data[4],
            }

    save_pkl('%s/location.pkl' % out_dir, loc_db)
