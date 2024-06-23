# Using Keras and Tensorflow you'll learn how to:
#
# create a fully-connected neural network architecture
# apply neural nets to two classic ML problems: regression and classification
# train neural nets with stochastic gradient descent, and
# improve performance with dropout, batch normalization, and other techniques

# So what is deep learning?
# Deep learning is an approach to machine learning characterized by deep stacks of computations.
# This depth of computation is what has enabled deep learning models to disentangle the kinds of complex and
# hierarchical patterns found in the most challenging real-world datasets.

# The Linear Unit
# as a formula  y=wx+b
# The input is x. Its connection to the neuron has a weight which is w.
# The b is a special kind of weight we call the bias.


# Setup plotting
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

import pandas as pd

red_wine = pd.read_csv('../dataset/red-wine.csv')
red_wine.head()

print(red_wine.shape)  # (rows, columns)

# Keras represents the weights of a neural network with tensors.
# Tensors are basically TensorFlow's version of a Numpy array with a few differences that make them better suited to deep learning.
# One of the most important is that tensors are compatible with GPU and TPU) accelerators.
# TPUs, in fact, are designed specifically for tensor computations.

# A model's weights are kept in its weights attribute as a list of tensors.
# Get the weights of the model you defined above.
# (If you want, you could display the weights with something like: print("Weights\n{}\n\nBias\n{}".format(w, b))).
from tensorflow import keras
from tensorflow.keras import layers

# YOUR CODE HERE
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[11])
])








