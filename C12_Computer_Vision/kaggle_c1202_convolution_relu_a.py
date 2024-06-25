# a convolutional classifier has two parts: a convolutional base and a head of dense layers.
# We learned that the job of the base is to extract visual features from an image,
# which the head would then use to classify the image.

# the two most important types of layers that you'll usually find in the base of a convolutional image classifier
# These are the convolutional layer with ReLU activation, and the maximum pooling layer

# The feature extraction performed by the base consists of three basic operations:
#
# Filter an image for a particular feature (convolution)
# Detect that feature within the filtered image (ReLU)
# Condense the image to enhance the features (maximum pooling)










import numpy as np
from itertools import product

def show_kernel(kernel, label=True, digits=None, text_size=28):
    # Format kernel
    kernel = np.array(kernel)
    if digits is not None:
        kernel = kernel.round(digits)

    # Plot kernel
    cmap = plt.get_cmap('Blues_r')
    plt.imshow(kernel, cmap=cmap)
    rows, cols = kernel.shape
    thresh = (kernel.max()+kernel.min())/2
    # Optionally, add value labels
    if label:
        for i, j in product(range(rows), range(cols)):
            val = kernel[i, j]
            color = cmap(0) if val > thresh else cmap(255)
            plt.text(j, i, val,
                     color=color, size=text_size,
                     horizontalalignment='center', verticalalignment='center')
    plt.xticks([])
    plt.yticks([])


from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3, activation='relu')
    # More layers follow
])


# Weight
# The weights a convnet learns during training are primarily contained in its convolutional layers.
# These weights we call kernels. We can represent them as small arrays:

# A kernel operates by scanning over an image and producing a weighted sum of pixel values.
# In this way, a kernel will act sort of like a polarized lens, emphasizing or deemphasizing certain patterns of information.

# Kernels define how a convolutional layer is connected to the layer that follows.
# The kernel above will connect each neuron in the output to nine neurons in the input.
# By setting the dimensions of the kernels with kernel_size, you are telling the convnet how to form these connections.
# Most often, a kernel will have odd-numbered dimensions -- like kernel_size=(3, 3) or (5, 5) --
# so that a single pixel sits at the center, but this is not a requirement.

# The kernels in a convolutional layer determine what kinds of features it creates.
# During training, a convnet tries to learn what features it needs to solve the classification problem.
# This means finding the best values for its kernels.


# Activations
# The activations in the network we call feature maps.
# They are what result when we apply a filter to an image; they contain the visual features the kernel extracts.
# Here are a few kernels pictured with feature maps they produced.

# From the pattern of numbers in the kernel, you can tell the kinds of feature maps it creates. Generally,
# what a convolution accentuates in its inputs will match the shape of the positive numbers in the kernel.
# The left and middle kernels above will both filter for horizontal shapes.
# With the filters parameter, you tell the convolutional layer how many feature maps you want it to create as output.


# Detect with ReLU
# After filtering, the feature maps pass through the activation function. The rectifier function has a graph like this:

# A neuron with a rectifier attached is called a rectified linear unit.
# For that reason, we might also call the rectifier function the ReLU activation or even the ReLU function.
# The ReLU activation can be defined in its own Activation layer, but most often you'll just include it as the activation function of Conv2D.

# Important
# You could think about the activation function as scoring pixel values according to some measure of importance.
# The ReLU activation says that negative values are not important and so sets them to 0.
# ("Everything unimportant is equally unimportant.")


import tensorflow as tf
import matplotlib.pyplot as plt
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

image_path = '../input/computer-vision-resources/car_feature.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.show();


# For the filtering step, we'll define a kernel and then apply it with the convolution.
# The kernel in this case is an "edge detection" kernel.
# You can define it with tf.constant just like you'd define an array in Numpy with np.array.
# This creates a tensor of the sort TensorFlow uses.


import tensorflow as tf

kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

plt.figure(figsize=(3, 3))
show_kernel(kernel)

# TensorFlow includes many common operations performed by neural networks in its tf.nn module.
# The two that we'll use are conv2d and relu. These are simply function versions of Keras layers.
#
# This next hidden cell does some reformatting to make things compatible with TensorFlow.
# The details aren't important for this example.
# Reformat for batch compatibility.
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)


image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    # we'll talk about these two in lesson 4!
    strides=1,
    padding='SAME',
)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.show();


# Next is the detection step with the ReLU function.
# This function is much simpler than the convolution, as it doesn't have any parameters to set.

image_detect = tf.nn.relu(image_filter)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.show();

# And now we've created a feature map! Images like these are what the head uses to solve its classification problem.
# We can imagine that certain features might be more characteristic of Cars and others more characteristic of Trucks.
# The task of a convnet during training is to create kernels that can find those features.

