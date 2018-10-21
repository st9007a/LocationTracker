#!/usr/bin/env python3
import tensorflow as tf

class Optimizer():

    def __init__(self, lr, tf_optimizer, decay=None):

        self.lr = lr
        self.tf_optimizer = tf_optimizer
        self.decay = decay

    def build(self, loss):

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = self.lr

        if self.decay is not None:
            self.learning_rate = tf.train.exponential_decay(self.lr, global_step=self.global_step,
                                                            decay_steps=10, decay_rate=self.decay,
                                                            staircase=True)

        self.optimizer = self.tf_optimizer(learning_rate=self.learning_rate)

        return self.optimizer.minimize(loss, global_step=self.global_step)
