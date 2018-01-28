import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from my_practice import utils

DATA_FILE = '../examples/data/birth_life_2010.txt'

# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# Remember both X and Y are scalars with type float
X = tf.placeholder(dtype=tf.float32, name='X')
Y = tf.placeholder(dtype=tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0.0
# Make sure to use tf.get_variable
w = tf.get_variable(name='weights', initializer=tf.constant(0.0))
b = tf.get_variable(name='bias', initializer=tf.constant(0.0))

# Step 4: build model to predict Y
# e.g. how would you derive at Y_predicted given X, w, and b
Y_predicted = tf.add(tf.multiply(w, X), b)

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()

# Create a filewriter to write the model's graph to TensorBoard
writer = tf.summary.FileWriter('./graphs/linreg_starter', tf.get_default_graph())

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run([w.initializer, b.initializer])

    # Step 8: train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, loss_val = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += loss_val

        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))
    writer.close()

    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])

print('Took: %f seconds' % (time.time() - start))

# plot the results
plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()