import numpy as np
from pypropack import svdp

from scipy.sparse import csr_matrix

np.random.seed(0)

# Create a random matrix
A = np.random.random((10, 20))

# compute SVD via propack and lapack
u, sigma, v = svdp(csr_matrix(A), 3)
u1, sigma1, v1 = np.linalg.svd(A, full_matrices=False)

# print the results
np.set_printoptions(suppress=True, precision=8)
print np.dot(u.T, u1)
print

print sigma
print sigma1
print

print np.dot(v, v1.T)
