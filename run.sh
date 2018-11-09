#!/usr/bin/env bash
# Setup environment
. config_env

# Parse raw data
python3 parser/location.py
python3 parser/tag2class.py
python3 parser/categorical.py
python3 parser/group.py
python3 parser/graph.py
python3 parser/user_checkins.py

# Train model
python3 train.py $1

# Generate result
# python3 eval.py
