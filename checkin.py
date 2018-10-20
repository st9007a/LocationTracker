#!/usr/bin/env python3
import pickle

import numpy as np

from utils.io import save_pkl

if __name__ == '__main__':

    user_checkin = {}

    with open('raw/checkins_missing.txt', 'r') as f:
        for line in f:

            user, series = line.rstrip('\n').split(':')

            if user not in user_checkin:
                user_checkin[user] = []

            series = series.split(',')
            struct_seq = []

            for i in range(0, len(series), 2):
                loc_id = series[i + 1] if series[i + 1] != '?' else user + '-?'
                struct_seq.append((int(series[i]), loc_id))

            user_checkin[user].append(struct_seq)

    save_pkl(user_checkin, 'output/user.checkin.pkl')
