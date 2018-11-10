#!/usr/bin/env python3
import numpy as np

from utils.io import read_pkl, save_pkl

if __name__ == '__main__':

    loc = read_pkl('tmp/location.pkl')
    user_miss = read_pkl('tmp/user_miss_pair.pkl')

    user_miss_loc = {}

    with open('raw/checkins_missing.txt', 'r') as f:
        for line in f:
            user, checkins = line.rstrip('\n').split(':')

            checkins = checkins.split(',')
            checkins = [(int(checkins[i]), checkins[i + 1])for i in range(0, len(checkins), 2)]
            # checkins = [el for el in checkins if el[1] == '?' or loc[el[1]]['country'] == 'US']

            for i, checkin in enumerate(checkins):
                if checkin[1] != '?':
                    continue

                if user not in user_miss_loc:
                    user_miss_loc[user] = []

                if i != 0 and checkins[i - 1][1] != '?':
                    user_miss_loc[user].append((loc[checkins[i - 1][1]]['lat'], loc[checkins[i - 1][1]]['lon']))

                if i != len(checkins) - 1 and checkins[i + 1][1] != '?':
                    user_miss_loc[user].append((loc[checkins[i + 1][1]]['lat'], loc[checkins[i + 1][1]]['lon']))

    save_pkl('tmp/user_miss_loc.pkl', user_miss_loc)
