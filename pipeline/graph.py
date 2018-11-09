#!/usr/bin/env python3
import numpy as np

from scipy import sparse
from sklearn.preprocessing import normalize

from utils.io import read_pkl, save_pkl

loc_db = read_pkl('tmp/location.pkl')
candidate = read_pkl('tmp/candidate.pkl')

def normalize_matrix(adj):
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -1.).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj)

def linear_law(time_delta):
    return (24 - time_delta) / 24

def power_law(time_delta, k=0.5):
    if time_delta == 0:
        return 1
    return time_delta ** (-k)

if __name__ == '__main__':

    # First pass: find all checkin place and user-miss pairs.
    counter = -1
    nodes = {}
    u_m_pair = set()

    with open('raw/checkins_missing.txt', 'r') as f:
        for line in f:
            user, checkins = line.rstrip('\n').split(':')

            checkins = checkins.split(',')
            checkins = [(int(checkins[i]), checkins[i + 1])for i in range(0, len(checkins), 2)]
            checkins = [el for el in checkins if el[1] == '?' or loc_db[el[1]]['country'] == 'US']

            for checkin in checkins:
                place = None
                if checkin[1] == '?':
                    place = user + '_?'
                    u_m_pair.add(place)

                else:
                    place = checkin[1]

                assert place is not None

                if place not in nodes:
                    counter += 1
                    nodes[place] = counter

    save_pkl('tmp/user_miss_pair.pkl', list(u_m_pair))
    graph_size = len(nodes.keys())

    node_list = [0] * graph_size
    for node in nodes:
        node_list[nodes[node]] = node

    save_pkl('tmp/nodes.pkl', node_list)

    # Second pass: build sparse matrix for graph.
    edges = {}

    with open('raw/checkins_missing.txt', 'r') as f:
        for line in f:
            user, checkins = line.rstrip('\n').split(':')

            checkins = checkins.split(',')
            checkins = [(int(checkins[i]), checkins[i + 1]) for i in range(0, len(checkins), 2)]
            checkins = [el for el in checkins if el[1] == '?' or loc_db[el[1]]['country'] == 'US']

            for i in range(len(checkins)):
                x_id = checkins[i][1] if checkins[i][1] != '?' else user + '_?'
                y_id = checkins[(i + 1) % len(checkins)][1] if checkins[(i + 1) % len(checkins)][1] != '?' else user + '_?'

                x_ind = nodes[x_id]
                y_ind = nodes[y_id]

                time = checkins[(i + 1) % len(checkins)][0] - checkins[i][0]
                if time < 0:
                    time += 24

                if x_id + '-' + y_id not in edges:
                    edges[x_id + '-' + y_id] = {'coord': (x_ind, y_ind), 'weight': 0}
                if y_id + '-' + x_id not in edges:
                    edges[y_id + '-' + x_id] = {'coord': (y_ind, x_ind), 'weight': 0}
                if x_id + '-' + x_id not in edges:
                    edges[x_id + '-' + x_id] = {'coord': (x_ind, x_ind), 'weight': 1}
                if y_id + '-' + y_id not in edges:
                    edges[y_id + '-' + y_id] = {'coord': (y_ind, y_ind), 'weight': 1}

                edges[x_id + '-' + y_id]['weight'] += power_law(time)

                if x_id != y_id:
                    edges[y_id + '-' + x_id]['weight'] += 0.1 * power_law(time)

    for node in node_list:
        if node + '-' + node not in edges:
            edges[node + '-' + node] = {'coord': (nodes[node], nodes[node]), 'weight': 1}

    edges = [edges[edge_id] for edge_id in edges]
    rows = [edge['coord'][0] for edge in edges]
    cols = [edge['coord'][1] for edge in edges]
    data = [edge['weight'] for edge in edges]
    print(graph_size)
    print(len(edges))

    graph = sparse.coo_matrix((data, (rows, cols)), shape=(graph_size, graph_size), dtype=np.float32)
    graph = normalize_matrix(graph)
    sparse.save_npz('tmp/graph.npz', graph)

    # Third pass: build node features
    time_features = np.zeros((graph_size, 24))
    location_features = np.zeros((graph_size, 6))
    with open('raw/checkins_missing.txt', 'r') as f:
        for line in f:
            user, checkins = line.rstrip('\n').split(':')

            checkins = checkins.split(',')
            checkins = [(int(checkins[i]), checkins[i + 1]) for i in range(0, len(checkins), 2)]
            checkins = [el for el in checkins if el[1] == '?' or loc_db[el[1]]['country'] == 'US']

            for i, checkin in enumerate(checkins):
                place = user + '_?' if checkin[1] == '?' else checkin[1]
                time_features[nodes[place], checkin[0]] += 1

                if i - 1 >= 0 and checkins[i - 1][1] != '?':
                    prev_place = checkins[i - 1][1]
                    location_features[nodes[place]][loc_db[prev_place]['group']] += 1

                if i + 1 < len(checkins) and checkins[i + 1][1] != '?':
                    next_place = checkins[i + 1][1]
                    location_features[nodes[place]][loc_db[next_place]['group']] += 1

    time_features = normalize(time_features)
    location_features = normalize(location_features)
    features = np.concatenate([time_features, location_features], axis=1)
    save_pkl('tmp/features.pkl', features)

    # Fourth pass: build node labels
    # labels = np.zeros((graph_size, len(categorical)))
    #
    # for node in nodes:
    #     if node not in u_m_pair:
    #         labels[nodes[node]][categorical.index(loc_db[node]['class'])] = 1
    #
    # print(np.sum(labels))
    # save_pkl('tmp/labels.pkl', labels)
