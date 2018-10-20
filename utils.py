#!/usr/bin/env python3
import math
import json

def distance(lat1, lon1, lat2, lon2):

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return c * 6371e3

def save_as_json(obj, file):
    with open(file, 'w') as j:
        json.dump(obj, j, indent=4)

def read_json(file):
    with open(file, 'r') as j:
        obj = json.load(j)

    return obj

if __name__ == '__main__':

    print(distance(40.733596, -74.003139, 40.758102, -73.975734))
