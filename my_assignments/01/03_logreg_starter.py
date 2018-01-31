import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time

from my_practice import utils

# Define parameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = '../../examples/data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)

# Create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)

img, label = iterator.get_next()
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stdev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
print("img.shape:{0}, label:{1}".format(img.shape, label.shape))
w = tf.get_variable(name='weights', shape=(img.shape[1], label.shape[1]),
                    initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
b = tf.get_variable(name='bias', shape=(1, label.shape[1]), initializer=tf.zeros_initializer())

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = tf.add(tf.matmul(img, w), b)

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')

# Step 6: define training op
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / n_test

writer = tf.summary.FileWriter('./graphs/logreg_starter', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)
    total_corrent_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_corrent_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_corrent_preds))
writer.close()
