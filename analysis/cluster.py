#!/usr/bin/env python3
import numpy as np
import pickle
import json

from sklearn.cluster import AgglomerativeClustering

if __name__ == '__main__':

    with open('src/db.pkl', 'rb') as p:
        db = pickle.load(p)

    with open('src/tags.pkl', 'rb') as p:
        tags = pickle.load(p)

    with open('src/loc.info.json', 'r') as j:
        info = json.load(j)

    users_poi = {}
    tag_idx = list(tags)

    with open('raw/checkins_missing.txt', 'r') as f:

        for line in f:
            user, seq = line.rstrip('\n').split(':')
            seq = seq.split(',')

            for i in range(1, len(seq), 2):
                if seq[i] == '?':
                    continue

                if db[seq[i]]['country'] != 'US':
                    continue

                if user not in users_poi:
                    users_poi[user] = []

                users_poi[user].append(seq[i])

    points = []
    users_list = []

    for user in users_poi:
        poi = [0] * info['tag_count']
        users_list.append(user)

        for lid in users_poi[user]:
            poi[tag_idx.index(db[lid]['tag'])] = 1

        points.append(poi)

    points = np.array(points)
    print(points.shape)

    model = AgglomerativeClustering(n_clusters=10)
    model.fit(points)

    for i in range(10):
        print(np.sum(np.where(model.labels_ == i, 1, 0)))

    nodes = [set(), set(), set(), set(), set(), set(), set(), set(), set(), set()]

    for i, user in enumerate(users_list):
        nodes[model.labels_[i]].update(users_poi[user])

    for i in range(10):
        print(len(nodes[i]))

