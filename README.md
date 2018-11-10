# Location Tracker

## Environment

- OS: Ubuntu 16.04
- Language: Python3.5

## Install

For gpu version:
1. Install cuda and cudnn. Their version should match tensorflow-gpu with version 1.8.0
2. Install python package: `pip3 install -r requires.txt`

For cpu version:
1. Install python package: `pip3 install -r requires.txt`
2. Remove tensorflow-gpu and install tensorflow: `pip3 uninstall tensorflow-gpu & pip3 install tensorflow==1.8.0`

## Run program

1. Generate all temporary file: `./build.sh`
  - It will generate a directory named `tmp` which contains all temporary files.
2. Train model: `./train.py my_model`
  - It will spend about 25 minutes to train a model with 5-fold cross validation in a nvidia 1080ti GPU. 
3. Generate result: `./eval.py my_model`
  - It generate a directory named `result`. And there are two files in the directory.
    - result.txt: output file of this homework
    - check.txt: the file that show the sorted classification probability of each user
