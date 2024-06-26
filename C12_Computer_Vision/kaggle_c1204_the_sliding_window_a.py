# Introduction
# In the previous two lessons, we learned about the three operations that carry out feature extraction from an image:
#
# filter with a convolution layer
# detect with ReLU activation
# condense with a maximum pooling layer

# The convolution and pooling operations share a common feature:
# they are both performed over a sliding window. With convolution, this "window" is given by the dimensions of the kernel,
# the parameter kernel_size. With pooling, it is the pooling window, given by pool_size.

# There are two additional parameters affecting both convolution and pooling layers --
# these are the strides of the window and whether to use padding at the image edges.
# The strides parameter says how far the window should move at each step,
# and the padding parameter describes how we handle the pixels at the edges of the input.

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  activation='relu'),
    layers.MaxPool2D(pool_size=2,
                     strides=1,
                     padding='same')
    # More layers follow
])

# Stride
# The distance the window moves at each step is called the stride.
# We need to specify the stride in both dimensions of the image: one for moving left to right and one for moving top to bottom.
# This animation shows strides=(2, 2), a movement of 2 pixels each step.


import tensorflow as tf
import matplotlib.pyplot as plt

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

image = circle([64, 64], val=1.0, r_shrink=3)
image = tf.reshape(image, [*image.shape, 1])
# Bottom sobel
kernel = tf.constant(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]],
)

show_kernel(kernel)



show_extraction(
    image, kernel,

    # Window parameters
    conv_stride=1,
    pool_size=2,
    pool_stride=2,

    subplot_shape=(1, 4),
    figsize=(14, 6),
)

# a model will use a convolution with a larger stride in it's initial layer.
# This will usually be coupled with a larger kernel as well. The ResNet50 model, for instance, uses  7×7
#   kernels with strides of 2 in its first layer.
#   This seems to accelerate the production of large-scale features without the sacrifice of too much information from the input.





















