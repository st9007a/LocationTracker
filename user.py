#!/usr/bin/env python3
import json
import pickle

import numpy as np
from scipy import sparse

with open('output/db.pkl', 'rb') as p:
    db = pickle.load(p)

def save_as_json(obj, file):
    with open(file, 'w') as j:
        json.dump(obj, j, indent=4)

def check_series(series):
    for i in range(1, len(series), 2):
        if series[i] == '?':
            continue

        if db[series[i]]['country'] != 'US':
            return False

    return True

if __name__ == '__main__':


    with open('raw/candidate_100_places.txt', 'r') as f:
        locs = f.readlines()
        locs = [loc.rstrip('\n') for loc in locs]

    save_as_json(locs, 'output/candidate.json')

    user_locs = set()
    with open('raw/checkins_missing.txt', 'r') as f:
        for line in f:
            user, series = line.rstrip('\n').split(':')
            series = series.split(',')

            if not check_series(series):
                continue

            for i in range(1, len(series), 2):
                if series[i] == '?':
                    continue

                user_locs.add(series[i])

    user_locs.update(locs)
    user_locs = list(user_locs)

    save_as_json(user_locs, 'output/user.loc.json')

    tags = set()
    for lid in user_locs:
        tags.add(db[lid]['tag'])

    save_as_json(list(tags), 'output/tag.json')

    user_rank = []
    users = []
    with open('raw/checkins_missing.txt', 'r') as f:
        for line in f:
            user, series = line.rstrip('\n').split(':')
            series = series.split(',')

            if not check_series(series):
                continue

            if len(users) == 0 or users[-1] != user:
                user_rank.append([0] * len(user_locs))
                users.append(user)

            for i in range(1, len(series), 2):
                if series[i] == '?':
                    continue

                user_rank[-1][user_locs.index(series[i])] += 1

    user_rank = sparse.csr_matrix(user_rank)
    sparse.save_npz('output/user.rank.npz', user_rank)
    save_as_json(users, 'output/user.json')
