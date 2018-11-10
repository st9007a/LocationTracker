#!/usr/bin/env python3
import numpy as np

from utils.io import read_pkl, save_pkl

if __name__ == '__main__':

    loc_db = read_pkl('tmp/location.pkl')
    user_checkins = {}

    c = 0
    with open('raw/checkins_missing.txt', 'r') as f:
        for line in f:
            user, checkins = line.rstrip('\n').split(':')

            checkins = checkins.split(',')
            checkins = [(int(checkins[i]), checkins[i + 1])for i in range(0, len(checkins), 2)]
            checkins = [el for el in checkins if el[1] in loc_db]
            # checkins = [el for el in checkins if el[1] in loc_db and loc_db[el[1]]['country'] == 'US']

            if user not in user_checkins:
                user_checkins[user] = set()

            for checkin in checkins:
                user_checkins[user].add(checkin[1])

    save_pkl('tmp/user_checkins.pkl', user_checkins)
