# A dense array, in the context of data analysis and machine learning,
# refers to a data structure where all elements are explicitly stored in memory,
# as opposed to a sparse array where only non-zero elements are stored to save memory.
# Dense arrays are straightforward to work with and are commonly used when the majority of elements in an array are non-zero or
# when simplicity and direct access to all elements are prioritized over memory efficiency.

import numpy as np

# Creating a dense array
# Dense arrays store every element, whether zero or not.
dense_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Dense Array:")
print(dense_array)

# Shape and memory usage of the dense array
print("\nShape of the dense array:", dense_array.shape)
print("Memory usage of the dense array:", dense_array.nbytes, "bytes")







