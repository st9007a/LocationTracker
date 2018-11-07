#!/usr/bin/env python3
from utils.io import read_pkl, save_pkl

if __name__ == '__main__':
    in_file = 'manual/class.txt'


    tag2class = {}
    ptr = -1

    with open(in_file, 'r') as f:
        for line in f:
            if line[0] != ' ':
                ptr += 1
                continue

            tag2class[line[2:-1]] = ptr

    save_pkl('tmp/tag2class.pkl', tag2class)

    with open('tmp/numclasses.txt', 'w') as f:
        f.write(str(ptr + 1) + '\n')
