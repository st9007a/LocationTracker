#!/usr/bin/env python3
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from utils.io import read_json, save_json, read_pkl

if __name__ == '__main__':

    users = read_json('output/user.json')
    user_miss = read_json('output/user.miss.json')
    user_rank = sparse.load_npz('output/user.rank.npz')
    user_checkin = read_pkl('output/user.checkin.pkl')
    loc_db = read_pkl('output/db.pkl')

    sim = cosine_similarity(user_rank)

    nodes = set()

    for i, score in enumerate(sim[users.index(user_miss[0])]):
        if score < 0.1:
            continue

        user = users[i]

        for series in user_checkin[user]:
            for ts in series:
                if ts[1] == '?' or loc_db[ts[1]]['country'] != 'US':
                    continue

                nodes.add(ts[1])

    print(len(nodes))
