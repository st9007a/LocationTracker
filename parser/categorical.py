#!/usr/bin/env python3
import numpy as np

from utils.io import read_pkl, save_pkl

if __name__ == '__main__':

    loc_db = read_pkl('tmp/location.pkl')

    candidate = {}
    categorical = set()
    with open('raw/candidate_100_places.txt', 'r') as f:
        lines = f.readlines()
        lines = [el.rstrip('\n') for el in lines]

    for place in lines:
        candidate[place] = loc_db[place]
        categorical.add(candidate[place]['tag'])

    with open('raw/checkins_missing.txt', 'r') as f:
        for line in f:
            user, checkins = line.rstrip('\n').split(':')

            checkins = checkins.split(',')
            checkins = [(int(checkins[i]), checkins[i + 1])for i in range(0, len(checkins), 2)]

            for checkin in checkins:
                if checkin[1] == '?' or loc_db[checkin[1]]['country'] != 'US':
                    continue

                categorical.add(loc_db[checkin[1]]['tag'])

    categorical = list(categorical)
    print(categorical)
    print(len(categorical))

    save_pkl('tmp/categorical.pkl', categorical)
    save_pkl('tmp/candidate.pkl', candidate)
