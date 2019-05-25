from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import glob
import tensorflow as tf
from matplotlib import pyplot as plt
tf.enable_eager_execution()

@tf.function
def simple_nn_layer(x, y):
  return tf.nn.relu(tf.matmul(x, y))


plt.subplot()
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

print(simple_nn_layer(x, y))
print(simple_nn_layer)