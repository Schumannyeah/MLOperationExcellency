# A sparse matrix is a matrix in which most of the elements are zero.
# Storing only the non-zero elements can save a significant amount of memory, especially for large matrices.
# In Python, the scipy.sparse module provides functionalities to handle sparse matrices efficiently.
# Let's illustrate creating and working with a sparse matrix using an example:

import numpy as np
from scipy.sparse import coo_matrix

# Example data: a 5x5 matrix with mostly zeros
row = np.array([0, 3, 1, 0])
col = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])  # Non-zero elements

# Creating a sparse matrix in COO (Coordinate List) format
sparse_matrix = coo_matrix((data, (row, col)), shape=(5, 5))

# Convert to other sparse formats (e.g., CSR, CSC) if needed for specific operations
sparse_matrix_csr = sparse_matrix.tocsr()
sparse_matrix_csc = sparse_matrix.tocsc()

print("Sparse Matrix (COO format):")
print(sparse_matrix)

print("\nSparse Matrix (CSR format):")
print(sparse_matrix_csr.toarray())

print("\nSparse Matrix (CSC format):")
print(sparse_matrix_csc.toarray())






