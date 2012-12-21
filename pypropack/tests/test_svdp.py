import numpy as np
from numpy.testing import assert_allclose
from pypropack import svdp, svdp_irl

from scipy.sparse import csr_matrix, csc_matrix

CONSTRUCTORS = {'array':np.array,
                'csr_matrix':csr_matrix,
                'csc_matrix':csc_matrix}

DTYPES = ['f', 'd', 'F', 'D']

WHICHES = ['L', 'S']

RTOLS = {'f':1E-5,
         'd':1E-10,
         'F':1E-5,
         'D':1E-10}

ATOLS = {'f':1E-5,
         'd':1E-10,
         'F':1E-5,
         'D':1E-10}

def is_complex_type(dtype):
    return np.dtype(dtype).char.isupper()


def generate_sparse_matrix(constructor, n, m, f,
                           dtype=float, rseed=0, **kwargs):
    rng = np.random.RandomState(rseed)
    if is_complex_type(dtype):
        M = (rng.rand(n, m) + 1j * rng.rand(n, m)).astype(dtype)
    else:
        M = rng.rand(n, m).astype(dtype)
    M[M.real > f] = 0
    return constructor(M, **kwargs)


def assert_orthogonal(u1, u2, rtol, atol):
    """Check that the first k rows of u1 and u2 are orthogonal"""
    I = abs(np.dot(u1.conj().T, u2))
    assert_allclose(I, np.eye(u1.shape[1], u2.shape[1]), rtol, atol)


def check_svdp(n, m, constructor, dtype, k, f=0.6, **kwargs):
    rtol = RTOLS[dtype]
    atol = ATOLS[dtype]

    M = generate_sparse_matrix(np.asarray, n, m, f, dtype)
    Msp = CONSTRUCTORS[constructor](M)

    u1, sigma1, v1 = np.linalg.svd(M, full_matrices=False)
    u2, sigma2, v2 = svdp(Msp, k=k, tol=rtol)

    # check that singular values agree
    assert_allclose(sigma1[:k], sigma2, rtol, atol)

    # check that singular vectors are orthogonal
    assert_orthogonal(u1, u2, rtol, atol)
    assert_orthogonal(v1.T, v2.T, rtol, atol)


def test_svdp(n=20, m=10):
    k = 5
    for constructor in CONSTRUCTORS:
        for dtype in DTYPES:
            yield (check_svdp, n, m, constructor, dtype, k)


def check_svdp_irl(n, m, constructor, dtype, k, which, f=0.6, **kwargs):
    rtol = RTOLS[dtype]
    atol = ATOLS[dtype]

    M = generate_sparse_matrix(np.asarray, n, m, f, dtype)
    Msp = CONSTRUCTORS[constructor](M)

    u1, sigma1, vt1 = np.linalg.svd(M, full_matrices=False)
    u2, sigma2, vt2 = svdp_irl(Msp, k=k, tol=rtol, which=which)

    if which.upper() == 'S':
        u1 = np.roll(u1, k, 1)
        vt1 = np.roll(vt1, k, 0)
        sigma1 = np.roll(sigma1, k)
    elif which.upper() == 'L':
        pass
    else:
        raise ValueError("which = '%s' not recognized")

    np.set_printoptions(precision=5, suppress=True)

    # check that singular values agree
    assert_allclose(sigma1[:k], sigma2, rtol, atol)

    # check that singular vectors are orthogonal
    assert_orthogonal(u1, u2, rtol, atol)
    assert_orthogonal(vt1.T, vt2.T, rtol, atol)


def test_svdp_irl(n=20, m=10):
    k = 5
    for constructor in CONSTRUCTORS:
        for dtype in DTYPES:
            for which in WHICHES:
                yield (check_svdp_irl, n, m,
                       constructor, dtype, k, which)
