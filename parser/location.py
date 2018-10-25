#!/usr/bin/env python3
import os

import numpy as np
from sklearn.cluster import KMeans

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

    loc_id_coord = [(k, loc_db[k]['lat'], loc_db[k]['lon']) for k in loc_db if loc_db[k]['country'] == 'US']
    loc_coord = [[k[1], k[2]] for k in loc_id_coord]
    loc_coord = np.array(loc_coord)

    cluster = KMeans(n_clusters=6)
    cluster.fit(loc_coord)

    for i in range(len(loc_coord)):
        loc_name = loc_id_coord[i][0]
        group_id = cluster.labels_[i]
        loc_db[loc_name]['group'] = group_id

    save_pkl('%s/location.pkl' % out_dir, loc_db)
