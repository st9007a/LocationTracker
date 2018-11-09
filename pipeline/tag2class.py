#!/usr/bin/env python3
from utils.io import read_pkl, save_pkl

if __name__ == '__main__':
    in_file = 'manual/reduce.txt'


    tag2class = {}
    target = ''

    with open(in_file, 'r') as f:
        for line in f:
            if line[0] != ' ':
                target = line.rstrip('\n').split(':')[1]
                continue

            tag2class[line[2:-1]] = target

    print(len(tag2class))
    save_pkl('tmp/tag2class.pkl', tag2class)
