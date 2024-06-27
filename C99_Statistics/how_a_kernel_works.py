# What is a Kernel?
# a kernel (also known as a filter or convolution matrix) is a small matrix used to apply effects
# like blurring, sharpening, edge detection, and more to an image.
# It is a crucial element in convolutional operations,
# which are fundamental in many image processing techniques and convolutional neural networks (CNNs).

# How a Kernel Works
# 1. Structure: A kernel is typically a small, square matrix (e.g., 3x3, 5x5) of numbers.
# 2. Convolution Operation: The kernel is applied to an image through a process called convolution.
#    This involves placing the kernel at each pixel of the image, multiplying the kernel values by the pixel values it overlaps,
#    and summing these products to get a new pixel value. This process is repeated for each pixel in the image.
# 3. Sliding Window: The kernel "slides" over the image, moving one pixel at a time (or more, depending on the stride),
#    and performs the convolution operation at each position.

# Edge Detection Kernel:
#
# Used to highlight edges in an image by detecting changes in intensity.
# Example: Sobel Kernel for vertical edges
# css
# 复制代码
# [-1, 0, 1]
# [-2, 0, 2]
# [-1, 0, 1]

# Blurring Kernel:
#
# Used to smooth an image, reducing noise and detail.
# Example: Gaussian Blur Kernel (3x3)
# csharp
# 复制代码
# [1/16, 1/8, 1/16]
# [1/8,  1/4, 1/8 ]
# [1/16, 1/8, 1/16]

# Sharpening Kernel:
#
# Used to enhance the edges and fine details in an image.
# Example: Sharpen Kernel
# css
# 复制代码
# [ 0, -1,  0]
# [-1,  5, -1]
# [ 0, -1,  0]

# Emboss Kernel:
#
# Used to give an image a 3D shadow effect, making it look like it has been embossed.
# Example: Emboss Kernel
# css
# 复制代码
# [-2, -1,  0]
# [-1,  1,  1]
# [ 0,  1,  2]

# Applications of Kernels
# 1. Feature Extraction: Kernels are used to extract important features from images, such as edges, corners, and textures.
# 2. Image Enhancement: Kernels can enhance certain aspects of an image, like sharpening details or smoothing noise.
# 3. Object Detection: In machine learning, kernels help in identifying objects by detecting relevant features.
# 4. Image Transformation: Kernels can alter images to achieve various effects, such as blurring, embossing, or edge detection.

# Visualization
# To better understand how a kernel affects an image, consider visualizing the original image and the image after applying the kernel.
# For example, applying an edge detection kernel to an image highlights the edges by increasing the contrast between adjacent pixel values.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('../input/computer-vision-resources/car_illus.jpg', 0)  # 0 for grayscale

# Define a kernel (e.g., Sobel kernel for edge detection)
sobel_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

# Apply the kernel to the image using filter2D
#  -1 This parameter specifies the desired depth of the output image.
#  The value -1 indicates that the output image will have the same depth as the input image.
#  Depth refers to the number of bits used to represent each pixel (e.g., 8-bit, 16-bit).
edge_detected_image = cv2.filter2D(image, -1, sobel_kernel)

# Display the original and processed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge_detected_image, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')

plt.show()

