# A "layer" in Keras is a very general kind of thing.
# A layer can be, essentially, any kind of data transformation.
# Many layers, like the convolutional and recurrent layers, transform data through use of neurons and differ primarily in the pattern of connections they form.
# Others though are used for feature engineering or just simple arithmetic. There's a whole world of layers to discover -- check them out!

# Without activation functions, neural networks can only learn linear relationships. In order to fit curves, we'll need to use activation functions.
# An activation function is simply some function we apply to each of a layer's outputs (its activations). The most common is the rectifier function  max(0,x)

# When we attach the rectifier to a linear unit, we get a rectified linear unit or ReLU.
# (For this reason, it's common to call the rectifier function the "ReLU function".)
# Applying a ReLU activation to a linear unit means the output becomes max(0, w * x + b), which we might draw in a diagram like:

# Stacking Dense Layers
# Now that we have some nonlinearity, let's see how we can stack layers to get complex data transformations.
# The layers before the output layer are sometimes called hidden since we never see their outputs directly.
#
# Now, notice that the final (output) layer is a linear unit (meaning, no activation function).
# That makes this network appropriate to a regression task, where we are trying to predict some arbitrary numeric value.
# Other tasks (like classification) might require an activation function on the output.

import tensorflow as tf

# Setup plotting
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

import pandas as pd

concrete = pd.read_csv('../dataset/concrete.csv')
print(concrete.head())

from tensorflow import keras
from tensorflow.keras import layers

input_shape = [8]

# YOUR CODE HERE
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

# or to be rewritten by a separated activation layers
# model = keras.Sequential([
#     layers.Dense(32, input_shape=[8]),
#     layers.Activation('relu'),
#     layers.Dense(32),
#     layers.Activation('relu'),
#     layers.Dense(1),
# ])

# Alternatives to ReLU
# There is a whole family of variants of the 'relu' activation -- 'elu', 'selu', and 'swish', among others
# -- all of which you can use in Keras. Sometimes one activation will perform better than another on a given task,
# so you could consider experimenting with activations as you develop a model. The ReLU activation tends to do well on most problems,
# so it's a good one to start with.

# YOUR CODE HERE: Change 'relu' to 'elu', 'selu', 'swish'... or something else
# ELU, short for Exponential Linear Unit, is an activation function introduced as an improvement over ReLU (Rectified Linear Unit) in deep learning models.
# It addresses the problem of dead neurons, which can occur when using ReLU, by allowing the activation of those neurons that produce negative outputs.
# SELU, which stands for Scaled Exponential Linear Unit, is another activation function designed for deep learning models,
# particularly for self-normalizing neural networks.
# It was introduced to address the challenges of training very deep networks by enabling them to maintain a stable distribution of activations
# throughout the layers.
# Swish is a novel activation function introduced by Google researchers in 2017.
# It is a self-gated activation function, designed to improve upon popular functions like ReLU by introducing a smooth,
# non-monotonic behavior that can enhance learning capabilities in deep neural networks. The mathematical formula for Swish is:
# f(x)=x⋅σ(βx)
# where 𝑥 is the input, and 𝜎 represents the sigmoid function
# which is applied to 𝛽𝑥 (with β typically set to 1 for most applications).
# The sigmoid function itself maps its input to a value between 0 and 1, acting as a gate to modulate the input 𝑥.
activation_layer = layers.Activation('swish')


# Generating Input Data: x = tf.linspace(-3.0, 3.0, 100) generates an array of 100 evenly spaced numbers between -3.0 and 3.0.
# This will serve as the input to our activation function.
x = tf.linspace(-3.0, 3.0, 100)
# Applying the Activation Function: By calling y = activation_layer(x),
# you're passing the generated input x through the ReLU activation function.
# ReLU is defined as f(x) = max(0, x), which means any negative input value is set to zero, while positive values are left unchanged.
y = activation_layer(x) # once created, a layer is callable just like a function

# plt.figure(dpi=100) sets the resolution of the figure
plt.figure(dpi=100)
# plt.plot(x, y) plots y (the output of ReLU) against x (the input values).
plt.plot(x, y)
# plt.xlim(-3, 3) sets the limits of the x-axis to match the input range.
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()






