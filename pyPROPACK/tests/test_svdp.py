import numpy as np
from numpy.testing import assert_allclose
from pyPROPACK import svdp


def simple_test_single(n=10, m=20, k=5):
    np.random.seed(0)
    A = np.random.random((n, m)).astype(np.float32)
    u1, s1, v1 = np.linalg.svd(A, full_matrices=False)
    u2, s2, v2 = svdp(A, k, tol=1E-5)

    # make sure singular vectors are orthogonal
    assert_allclose(abs(np.dot(u1[:, :k].T, u2)),
                    np.eye(k), rtol=1E-5, atol=1E-5)
    assert_allclose(abs(np.dot(v1[:k], v2.T)),
                    np.eye(k), rtol=1E-5, atol=1E-5)
    assert_allclose(s1[:k], s2, rtol=1E-5)


def simple_test_doubple(n=10, m=20, k=5):
    np.random.seed(0)
    A = np.random.random((n, m)).astype(np.float64)
    u1, s1, v1 = np.linalg.svd(A, full_matrices=False)
    u2, s2, v2 = svdp(A, k, tol=1E-10)

    # make sure singular vectors are orthogonal
    assert_allclose(abs(np.dot(u1[:, :k].T, u2)),
                    np.eye(k), rtol=1E-10, atol=1E-10)
    assert_allclose(abs(np.dot(v1[:k], v2.T)),
                    np.eye(k), rtol=1E-10, atol=1E-10)
    assert_allclose(s1[:k], s2, rtol=1E-10)
