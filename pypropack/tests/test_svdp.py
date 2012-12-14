import numpy as np
from numpy.testing import assert_allclose
from pypropack import svdp

from scipy.sparse import csr_matrix, csc_matrix

CONSTRUCTORS = [np.array, csr_matrix, csc_matrix]

DTYPES = ['f', 'd']

RTOLS = {'f':1E-5,
         'd':1E-10}

ATOLS = {'f':1E-5,
         'd':1E-10}


def generate_sparse_matrix(constructor, n, m, f,
                           dtype=float, rseed=0, **kwargs):
    rng = np.random.RandomState(rseed)
    M = rng.rand(n, m).astype(dtype)
    M[M > f] = 0
    return constructor(M, **kwargs)


def assert_orthogonal(u1, u2, k, rtol, atol):
    """Check that the first k rows of u1 and u2 are orthogonal"""
    I = abs(np.dot(u1[:, :k].T, u2[:, :k]))
    assert_allclose(I, np.eye(k), rtol, atol)


def check_svdp(n, m, constructor, dtype, k, f=0.6, **kwargs):
    rtol = RTOLS[dtype]
    atol = ATOLS[dtype]

    M = generate_sparse_matrix(np.asarray, n, m, f, dtype)
    Msp = constructor(M)

    u1, sigma1, v1 = np.linalg.svd(M, full_matrices=False)
    u2, sigma2, v2 = svdp(Msp, k=k, tol=rtol)

    # make sure singular vectors are orthogonal
    assert_orthogonal(u1, u2, k, rtol, atol)
    assert_orthogonal(v1.T, v2.T, k, rtol, atol)

    # make sure singular values agree
    assert_allclose(sigma1[:k], sigma2, rtol, atol)


def test_svdp(n=20, m=10):
    k = 5
    for constructor in CONSTRUCTORS:
        for dtype in DTYPES:
            yield (check_svdp, n, m, constructor, dtype, k)
