import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

from my_practice import utils

DATA_FILE = '../examples/data/birth_life_2010.txt'

# In order to use eager execution, `tfe.enable_eager_execution()` must be
# called at the very beginning of a TensorFlow program.
tfe.enable_eager_execution()

# Read the data into a dataset.
data, n_samples = utils.read_birth_life_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))

# Create weight and bias variables, initialized to 0.0
w = tfe.Variable(0.0)
b = tfe.Variable(0.0)
