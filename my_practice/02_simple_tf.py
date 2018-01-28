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
    # print(sess.run(tf.realdiv(b, a)))
    print("truncatediv:", sess.run(tf.truncatediv(b, a)))
    print("floor_div:", sess.run(tf.floor_div(b, a)))

# Example 3: multiplying tensors
a = tf.constant([10, 20], name='a')
b = tf.constant([2, 3], name='b')

with tf.Session() as sess:
    print("multiply:", sess.run(tf.multiply(a, b)))
    print("tensordot:", sess.run(tf.tensordot(a, b, 1)))

# Example 4: Python native type
t_0 = 19
x = tf.zeros_like(t_0)
y = tf.ones_like(t_0)

t_1 = ['apple', 'peach', 'banana']
x = tf.zeros_like(t_1)


t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]]
x = tf.zeros_like(t_2)
y = tf.ones_like(t_2)

print(tf.int32.as_numpy_dtype())

# Example 5: printing your graph's definition
my_const = tf.constant([1.0, 2.0], name='my_const')
print(tf.get_default_graph().as_graph_def())
