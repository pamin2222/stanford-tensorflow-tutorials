import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

i = tf.constant(0)
while i < 1000:
    i = tf.add(i, 1)
    print(i)
