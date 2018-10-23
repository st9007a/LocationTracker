#!/usr/bin/env python3
import tensorflow as tf

__training_mode = tf.placeholder(tf.bool, name='training')

def learning_phase():
    return __training_mode
