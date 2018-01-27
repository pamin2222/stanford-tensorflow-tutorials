import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Ex1: feed_dict with placeholder
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b

writer = tf.summary.FileWriter('graphs/placeholders', tf.get_default_graph())
with tf.Session() as sess:
    print("ex1:", sess.run(c, feed_dict={a: [1, 2, 3]}))
writer.close()

# Ex2 : feed_dict with variables
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
    print("ex2:", sess.run(b))
    print("ex2:", sess.run(b, feed_dict={a: 15}))
