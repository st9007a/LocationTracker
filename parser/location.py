#!/usr/bin/env python3
import os

import numpy as np

from utils.io import save_pkl

chars = 'abcdefghijklmnopqrstuvwxyz'

if __name__ == '__main__':
    in_file = 'raw/loc_id_info.txt'
    out_dir = 'output'

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    loc_db = {}

    with open(in_file, 'r') as f:

        for line in f:
            data = line.rstrip('\n').split('\t')
            loc_db[data[0]] = {
                'lat': float(data[1]),
                'lon': float(data[2]),
                'tag': ''.join([c for c in data[3].lower() if c in chars]),
                'country': data[4],
            }

    save_pkl('%s/location.pkl' % out_dir, loc_db)
