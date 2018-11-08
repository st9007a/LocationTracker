#!/usr/bin/env python3
import json
import pickle

def save_json(file, obj):
    with open(file, 'w') as j:
        json.dump(obj, j, indent=4)

def read_json(file):
    with open(file, 'r') as j:
        obj = json.load(j)

    return obj

def save_pkl(file, obj):
    print('save file:', file)
    with open(file, 'wb') as p:
        pickle.dump(obj, p, protocol=pickle.HIGHEST_PROTOCOL)

def read_pkl(file):
    with open(file, 'rb') as p:
        obj = pickle.load(p)

    return obj

if __name__ == '__main__':

    print(distance(40.733596, -74.003139, 40.758102, -73.975734))
