#!/usr/bin/env python3
import numpy as np

from sklearn.cluster import KMeans

from utils.io import read_pkl, save_pkl
from utils.location import distance

if __name__ == '__main__':

    loc_db = read_pkl('tmp/location.pkl')
    loc_in_checkins = {}

    with open('raw/checkins_missing.txt', 'r') as f:
        for line in f:
            user, checkins = line.rstrip('\n').split(':')

            checkins = checkins.split(',')
            checkins = [(int(checkins[i]), checkins[i + 1])for i in range(0, len(checkins), 2)]
            checkins = [el for el in checkins if el[1] != '?' and loc_db[el[1]]['country'] == 'US']

            for checkin in checkins:
                if checkin[1] not in loc_in_checkins:
                    loc_in_checkins[checkin[1]] = [loc_db[checkin[1]]['lat'], loc_db[checkin[1]]['lon']]

    candidate = read_pkl('tmp/candidate.pkl')

    for cand in candidate:
        if cand not in loc_in_checkins:
            loc_in_checkins[cand] = [loc_db[cand]['lat'], loc_db[cand]['lon']]

    loc_id_coord = [(k, loc_in_checkins[k]) for k in loc_in_checkins]
    loc_coord = [el[1] for el in loc_id_coord]

    loc_coord = np.array(loc_coord)

    cluster = KMeans(n_clusters=6)
    cluster.fit(loc_coord)

    for i in range(len(loc_coord)):
        loc_name = loc_id_coord[i][0]
        group_id = cluster.labels_[i]
        loc_db[loc_name]['group'] = group_id

    save_pkl('tmp/location.pkl', loc_db)
