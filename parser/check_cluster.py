#!/usr/bin/env python3
import numpy as np

from sklearn.cluster import KMeans

from utils.io import read_pkl, distance

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

    for i in range(2, 11):
        cluster = KMeans(n_clusters=i)
        cluster.fit(loc_coord)
        dist = [0] * i
        count = [0] * i
        for j in range(len(loc_coord)):
            dist[cluster.labels_[j]] += distance(cluster.cluster_centers_[cluster.labels_[j]][0],
                                                 cluster.cluster_centers_[cluster.labels_[j]][1],
                                                 loc_coord[j][0], loc_coord[j][1])
            count[cluster.labels_[j]] += 1
        for j in range(i):
            dist[j] /= count[j]
            dist[j] /= 1000
        avg_dist = sum(dist) / len(dist)
        print('Clusters: %d, Average Distance (km): %.6f' % (i, avg_dist))

