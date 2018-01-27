import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf


# Example 1: Simple ways to create log file writer
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

writer = tf.summary.FileWriter('./graphs/simple', tf.get_default_graph())
with tf.Session() as sess:
    print(sess.run(x))
writer.close()


# Example 2: The wonderful wizard of div
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')

with tf.Session() as sess:
    print("div:", sess.run(tf.div(b, a)))
    print("divide:", sess.run(tf.divide(b, a)))
    print("truediv:", sess.run(tf.truediv(b, a)))
    print("floordiv:", sess.run(tf.floordiv(b, a)))
    #print(sess.run(tf.realdiv(b, a)))
    print("truncatediv:", sess.run(tf.truncatediv(b, a)))
    print("floor_div:", sess.run(tf.floor_div(b, a)))