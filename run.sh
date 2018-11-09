#!/usr/bin/env bash
# Setup environment
. config_env

# Parse raw data
python3 pipeline/location.py
python3 pipeline/tag2class.py
python3 pipeline/setclass.py
python3 pipeline/group.py
python3 pipeline/graph.py
python3 pipeline/label.py
python3 pipeline/fix_group_features.py
python3 pipeline/user_checkins.py
python3 pipeline/user_miss_locate.py

# Train model
# python3 train.py $1

# Generate result
# python3 eval.py
