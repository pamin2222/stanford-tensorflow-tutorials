import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

s = tf.Variable(2, name='scalar')
m = tf.Variable([[0, 1], [2, 3]], name='matrix')
W = tf.Variable(tf.zeros([784, 10]), name='big_matrix')
V = tf.Variable(tf.truncated_normal([784, 10]), name='normal_matrix')

s = tf.get_variable('scalar', initializer=tf.constant(2))
m = tf.get_variable('matrix', initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable('big_matrix', shape=(784, 10), initializer=tf.zeros_initializer())
V = tf.get_variable('normal_matrix', shape=(784, 10), initializer=tf.truncated_normal_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(V.eval())

# Example 2: assigning values to variables
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W))

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(assign_op)
    print(W.eval())

# create a variable whose original value is 2
with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    a = tf.get_variable('scalar', initializer=tf.constant(2))
    a_times_two = a.assign(a * 2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("a_times_two:", sess.run(a_times_two))
        print("a_times_two:", sess.run(a_times_two))
        print("a_times_two:", sess.run(a_times_two))

W = tf.Variable(10)
with tf.Session() as sess:
    sess.run(W.initializer)
    print("assign_add:", sess.run(W.assign_add(10)))  # 20
    print("assign_sub:", sess.run(W.assign_sub(2)))  # 18

# Example 3: Each session has its own copy of variable
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10)))  # 20
print(sess2.run(W.assign_sub(2)))  # 8
print(sess1.run(W.assign_add(100)))  # 120
print(sess2.run(W.assign_sub(50)))  # -42
sess1.close()
sess2.close()
