#!/usr/bin/env python3
import pickle

if __name__ == '__main__':

    with open('src/db.pkl', 'rb') as p:
        db = pickle.load(p)

    country = {}

    with open('raw/candidate_100_places.txt', 'r') as f:

        for loc in f:
            loc = loc.rstrip('\n')

            if db[loc]['country'] not in country:
                country[db[loc]['country']] = 0
            country[db[loc]['country']] += 1

    print(country)
