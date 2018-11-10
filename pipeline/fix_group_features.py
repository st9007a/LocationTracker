#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import normalize

from utils.io import read_pkl, save_pkl

if __name__ == '__main__':

    user_checkins = read_pkl('tmp/user_checkins.pkl')
    loc_db = read_pkl('tmp/location.pkl')
    nodes = read_pkl('tmp/nodes.pkl')
    node_features = read_pkl('tmp/features.pkl')

    for i, node in enumerate(nodes):
        if node[-1] != '?':
            continue
        if np.sum(node_features[i][24:]) > 0:
            continue

        user = node[:-2]
        group_features = np.zeros((6, 1))

        for checkin in user_checkins[user]:
            if checkin in loc_db:
                g = loc_db[checkin]['group']
                group_features[g][0] += 1

        group_features = normalize(group_features, axis=0)

        for j in range(6):
            node_features[i][j + 24] = group_features[j]

    save_pkl('tmp/features', node_features)
