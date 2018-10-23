#!/usr/bin/env python3
import os

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from utils.io import read_json, save_json, read_pkl, save_pkl

users = read_json('output/user.json')
user_miss = read_json('output/user.miss.json')
user_rank = sparse.load_npz('output/user.rank.npz')
user_checkin = read_pkl('output/user.checkin.pkl')
loc_db = read_pkl('output/db.pkl')

def normalize_matrix(adj):
    rowsum = np.sum(adj, axis=1)
    d = np.power(rowsum, -1)
    d = np.diag(d)

    return np.matmul(d, adj)

def build_graph_matrix(nodes, user_group):
    graph_matrix = np.zeros((len(nodes), len(nodes)))

    for user in user_group:
        series = user_checkin[user]

        for day_series in series:
            day_series_clean = [el for el in day_series if el[1] not in loc_db or loc_db[el[1]]['country'] == 'US']

            for i in range(len(day_series_clean) - 1):
                x = nodes.index(day_series_clean[i][1])
                y = nodes.index(day_series_clean[(i + 1) % len(day_series_clean)][1])

                time = day_series_clean[(i + 1) % len(day_series_clean)][0] - day_series_clean[i][0]
                if time < 0:
                    time += 24

                graph_matrix[x][y] += (24 - time) / 24

                if x != y:
                    graph_matrix[y][x] += 0.1 * (24 - time) / 24

    graph_matrix += np.identity(len(nodes))

    return sparse.csr_matrix(normalize_matrix(graph_matrix), dtype=np.float32)

def build_node_features(nodes, user_gropu):
    node_features = np.zeros((len(nodes), 24))

    for user in user_group:
        series = user_checkin[user]

        for day_series in series:
            day_series_clean = [el for el in day_series if el[1] not in loc_db or loc_db[el[1]]['country'] == 'US']

            for ts in day_series_clean:
                node_features[nodes.index(ts[1])][ts[0]] += 1

    return normalize(node_features)

def build_node_labels(nodes):
    tags = set()

    for node in nodes:
        if node in loc_db:
            tags.add(loc_db[node]['tag'])

    tags = list(tags)
    labels = []
    test_idx = []

    for i, node in enumerate(nodes):
        label = [0] * len(tags)

        if node in loc_db:
            label[tags.index(loc_db[node]['tag'])] = 1
        else:
            test_idx.append(i)

        labels.append(label)

    return np.array(labels, dtype=int), test_idx

if __name__ == '__main__':


    sim = cosine_similarity(user_rank)

    nodes = set()
    user_group = set()
    count = 0

    for i, score in enumerate(sim[users.index(user_miss[0])]):
        if score < 0.1:
            continue

        if users[i] in user_miss:
            count += 1

        user = users[i]
        user_group.add(user)

        for series in user_checkin[user]:
            for ts in series:
                if ts[1] in loc_db and loc_db[ts[1]]['country'] != 'US':
                    continue

                nodes.add(ts[1])

    print('Node count:', len(nodes))
    print('Missing checkin users: ', count)

    nodes = list(nodes)
    user_group = list(user_group)

    graph_matrix = build_graph_matrix(nodes, user_group)
    node_features = build_node_features(nodes, user_group)
    node_labels, test_mask = build_node_labels(nodes)

    if not os.path.isdir('features'):
        os.makedirs('features')

    sparse.save_npz('features/graph.npz', graph_matrix)
    np.save('features/nodes.npy', node_features)
    np.save('features/labels.npy', node_labels)
    save_pkl('features/test.mask.pkl', test_mask)
