#!/usr/bin/env python3
import os
import pickle
import json

if __name__ == '__main__':

    if not os.path.isdir('output'):
        os.makedirs('output')

    db = {}
    tags = set()
    tags_us = set()
    count_us = 0

    with open('raw/loc_id_info.txt', 'r') as f:

        for line in f:
            data = line.rstrip('\n').split('\t')
            db[data[0]] = {
                'lat': float(data[1]),
                'lon': float(data[2]),
                'tag': data[3],
                'country': data[4],
            }

            tags.add(data[3])

            if data[4] == 'US':
                tags_us.add(data[3])
                count_us += 1

    with open('output/db.pkl', 'wb') as p:
        pickle.dump(db, p, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('output/tags.pkl', 'wb') as p:
    #     pickle.dump(tags_us, p, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # print(len(tags))
    # print(len(tags_us))
    # print(tags - tags_us)
    #
    # loc_info = {
    #     'exclude': list(tags - tags_us),
    #     'tag_count': len(tags_us),
    #     'count': len(db),
    #     'count_us_only': count_us,
    # }
    #
    # with open('output/loc.info.json', 'w') as j:
    #     json.dump(loc_info, j, indent=4)
