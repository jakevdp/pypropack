"""
pypropack: a Python wrapper for PROPACK
---------------------------------------

PROPACK is a collection of Fortran routines for iterative computation
of partial SVDs of large matrices or linear operators.
"""
# Author: Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD
#
# PROPACK source is BSD licensed, and available at
#   http://soi.stanford.edu/~rmunk/PROPACK/
#
# This python wrapper is BSD licensed, and available at
#   http://github.com/jakevdp/pypropack

__all__ = ['svdp', 'svdp_irl']

import numpy as np
from scipy.sparse.linalg import aslinearoperator

import _spropack
import _dpropack
import _cpropack
import _zpropack

_lansvd_dict = {'f': _spropack.slansvd,
                'd': _dpropack.dlansvd,
                'F': _cpropack.clansvd,
                'D': _zpropack.zlansvd}

_lansvd_irl_dict = {'f': _spropack.slansvd_irl,
                    'd': _dpropack.dlansvd_irl,
                    'F': _cpropack.clansvd_irl,
                    'D': _zpropack.zlansvd_irl}


class _AProd(object):
    """Wrapper class for linear operator

    The call signature of the __call__ method matches the callback of
    the PROPACK routines.
    """
    def __init__(self, A):
        try:
            self.A = aslinearoperator(A)
        except:
            self.A = aslinearoperator(np.asarray(A))

    def __call__(self, transa, m, n, x, y, sparm, iparm):
        if transa == 'n':
            y[:] = self.A.matvec(x)
        else:
            y[:] = self.A.rmatvec(x)

    @property
    def shape(self):
        return self.A.shape

    @property
    def dtype(self):
        try:
            return self.A.dtype
        except AttributeError:
            return A.matvec(np.zeros(A.shape[1])).dtype


def svdp(A, k=3, kmax=None, compute_u=True, compute_v=True,
         tol=0, v0=None, full_output=False):
    """Compute the singular value decomposition of A using PROPACK

    Parameters
    ----------
    A : array_like, sparse matrix, or LinearOperator
        Operator for which svd will be computed.  If A is a LinearOperator
        object, it must define both ``matvec`` and ``rmatvec`` methods.
    k : int
        number of singular values/vectors to compute
    kmax : int
        maximal number of iterations / maximal dimension of Krylov subspace.
        default is 10 * k
    compute_u : bool
        if True (default) then compute left singular vectors u
    compute_v : bool
        if True (default) then compute right singular vectors v
    tol : float (optional)
        The desired relative accuracy for computed singular values.
        If not specified, it will be set based on machine precision.
    v0 : starting vector (optional)
        Starting vector for iterations: should be of length A.shape[0].
        If not specified, PROPACK will generate a starting vector.
    full_output : boolean
        If True, then return info and sigma_bound.  Default is False

    Returns
    -------
    u : ndarray
        The top k left singular vectors, shape = (A.shape[0], 3),
        returned only if compute_u is True.
    sigma : ndarray
        The top k singular values, shape = (k,)
    vt : ndarray
        The top k right singular vectors, shape = (3, A.shape[1]),
        returned only if compute_v is True.
    info : integer
        convergence info, returned only if full_output is True
        - INFO = 0  : The K largest singular triplets were computed succesfully
        - INFO = J>0, J<K: An invariant subspace of dimension J was found.
        - INFO = -1 : K singular triplets did not converge within KMAX
                     iterations.   
    sigma_bound : ndarray
        the error bounds on the singular values sigma, returned only if
        full_output is True

    Examples
    --------
    >>> A = np.random.random((10, 20))
    >>> u, sigma, vt = svdp(A, 3)
    >>> np.set_printoptions(precision=3, suppress=True)
    >>> print abs(np.dot(u.T, u))
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    >>> print abs(np.dot(vt, vt.T))
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    """
    aprod = _AProd(A)
    typ = aprod.dtype.char

    # TODO: how to deal with integer types? up-cast?
    try:
        lansvd = _lansvd_dict[typ]
    except:
        raise ValueError("operator type '%s' not supported" % typ)

    m, n = aprod.shape

    if compute_u:
        jobu = 'y'
    else:
        jobu = 'n'

    if compute_v:
        jobv = 'y'
    else:
        jobv = 'n'

    if kmax is None:
        kmax = 10 * k

    # these will be the output arrays
    u = np.zeros((m, kmax + 1), order='F', dtype=typ)
    v = np.zeros((n, kmax), order='F', dtype=typ)

    # specify the starting vector.  If all zero, then it will be generated
    # automatically by PROPACK
    if v0 is not None:
        try:
            u[:, 0] = v0
        except:
            raise ValueError("v0 must be of length %i" % m)

    # options for the fit
    # TODO: document these options and make these parameters adjustable
    #       via keyword arguments to svdp()
    ioption = np.zeros(2, dtype='i')
    ioption[0] = 0  # controls Gram-Schmidt reorthogonalization
    ioption[1] = 1  # controls re-orthonormalization

    eps = 2 * np.finfo(typ).eps
    dsoption = np.zeros(3, dtype=typ.lower())
    dsoption[0] = eps ** 0.5   # level of orthogonality to maintain
    dsoption[1] = eps ** 0.75  # purge vectors larger than this
    dsoption[2] = 0.0          # estimate of ||A||

    # TODO: choose lwork based on inputs (see propack documentation)
    # this is the size needed if compute_u and compute_v are both True
    nb = 3  # number of blocks
    lwork = (m + n
             + 9 * kmax
             + 5 * kmax * kmax + 4
             + max(3 * kmax ** 2 + 4 * kmax + 4,
                   nb * max(m, n)))
    work = np.zeros(lwork, dtype='f')

    # TODO: choose liwork based on inputs (see propack documentation)
    # this is the size needed if compute_u and compute_v are both True
    liwork = 8 * kmax
    iwork = np.zeros(liwork, dtype='i')

    # dummy arguments: these are passed to aprod, and not used
    dparm = np.zeros(1, dtype='f')
    iparm = np.zeros(1, dtype='i')

    if typ in ['F', 'D']:
        cwork = np.zeros(lwork, dtype=typ)
        u, sigma, bnd, v, info = lansvd(jobu, jobv, m, n, k,
                                        aprod, u, v, tol, work, cwork, iwork,
                                        dsoption, ioption, dparm, iparm)
    else:
        u, sigma, bnd, v, info = lansvd(jobu, jobv, m, n, k,
                                        aprod, u, v, tol, work, iwork,
                                        dsoption, ioption, dparm, iparm)

    # construct return tuple
    ret = ()
    if compute_u:
        ret = ret + (u[:, :k],)
    ret = ret + (sigma,)
    if compute_v:
        ret = ret + (v[:, :k].conj().T,)

    if full_output:
        ret = ret + (info, bnd)

    return ret


def svdp_irl(A, k=3, which='L', kmax=None, shifts=3, maxiter=10,
             compute_u=True, compute_v=True, tol=0, v0=None,
             full_output=False):
    """Compute the svd of A using PROPACK

    svd_irl uses the iteratively restarted Lanczos algorithm

    Parameters
    ----------
    A : array_like, sparse matrix, or LinearOperator
        Operator for which svd will be computed.  If A is a LinearOperator
        object, it must define both ``matvec`` and ``rmatvec`` methods.
    k : int
        number of singular values/vectors to compute
    which : string
        which singluar triplets to compute:
        - 'L': compute triplets corresponding to the k largest singular values
        - 'S': compute triplets corresponding to the k smallest singular values
    kmax : int
        maximal dimension of Krylov subspace.
        default is 10 * k
    shifts : int
        number of shifts per restart.  Default is 3
    maxiter : int
        maximum number of restarts.  Default is 10
    compute_u : bool
        if True (default) then compute left singular vectors u
    compute_v : bool
        if True (default) then compute right singular vectors v
    tol : float (optional)
        The desired relative accuracy for computed singular values.
        If not specified, it will be set based on machine precision.
    v0 : starting vector (optional)
        Starting vector for iterations: should be of length A.shape[0].
        If not specified, PROPACK will generate a starting vector.
    full_output : boolean
        If True, then return info and sigma_bound.  Default is False

    Returns
    -------
    u : ndarray
        the top k left singular vectors, shape = (A.shape[0], 3)
    sigma : ndarray
        the top k singular values, shape = (k,)
    vt : ndarray
        the top k right singular vectors, shape = (3, A.shape[1])
    info : integer
        convergence info, returned only if full_output is True
        - INFO = 0  : The K largest singular triplets were computed succesfully
        - INFO = J>0, J<K: An invariant subspace of dimension J was found.
        - INFO = -1 : K singular triplets did not converge within KMAX
                     iterations.   
    sigma_bound : ndarray
        the error bounds on the singular values sigma, returned only if
        full_output is True

    Examples
    --------
    >>> A = np.random.random((10, 20))
    >>> u, sigma, vt = svdp_irl(A, 3, which='L')
    >>> np.set_printoptions(precision=3, suppress=True)
    >>> print abs(np.dot(u.T, u))
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    >>> print abs(np.dot(vt, vt.T))
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    """
    aprod = _AProd(A)
    typ = aprod.dtype.char

    # TODO: how to deal with integer types? up-cast?
    try:
        lansvd_irl = _lansvd_irl_dict[typ]
    except:
        raise ValueError("operator type '%s' not supported" % typ)

    m, n = aprod.shape

    if compute_u:
        jobu = 'y'
    else:
        jobu = 'n'

    if compute_v:
        jobv = 'y'
    else:
        jobv = 'n'

    which = which.lower()
    if which not in ['l', 's']:
        raise ValueError("`which` must be either 'L' or 'S'")

    maxiter = int(maxiter)
    if maxiter < 1:
        raise ValueError('`maxiter` must be positive')

    shifts = int(shifts)
    if shifts < 1:
        raise ValueError('`shifts` must be positive')

    if kmax is None:
        kmax = 10 * k

    # these will be the output arrays
    u = np.zeros((m, kmax + 1), order='F', dtype=typ)
    v = np.zeros((n, kmax), order='F', dtype=typ)

    # specify the starting vector.  If all zero, then it will be generated
    # automatically by PROPACK
    if v0 is not None:
        try:
            u[:, 0] = v0
        except:
            raise ValueError("v0 must be of length %i" % m)

    # options for the fit
    # TODO: document these options and make these parameters adjustable
    #       via keyword arguments to svdp()
    ioption = np.zeros(2, dtype='i')
    ioption[0] = 0  # controls Gram-Schmidt reorthogonalization
    ioption[1] = 1  # controls re-orthonormalization

    eps = 2 * np.finfo(typ).eps
    dsoption = np.zeros(4, dtype=typ.lower())
    dsoption[0] = eps ** 0.5   # level of orthogonality to maintain
    dsoption[1] = eps ** 0.75  # purge vectors larger than this
    dsoption[2] = 0.0          # estimate of ||A||
    dsoption[3] = 0.002

    # TODO: choose lwork based on inputs (see propack documentation)
    # this is the size needed if compute_u and compute_v are both True
    nb = 3  # number of blocks
    lwork = (m + n
             + 9 * kmax
             + 5 * kmax * kmax + 4
             + max(3 * kmax ** 2 + 4 * kmax + 4,
                   nb * max(m, n)))
    work = np.zeros(lwork, dtype='f')

    # TODO: choose liwork based on inputs (see propack documentation)
    # this is the size needed if compute_u and compute_v are both True
    liwork = 8 * kmax
    iwork = np.zeros(liwork, dtype='i')

    # dummy arguments: these are passed to aprod, and not used
    dparm = np.zeros(1, dtype=typ.lower())
    iparm = np.zeros(1, dtype='i')

    if typ in ['F', 'D']:
        # TODO: check that cwork is of the correct length
        cwork = np.zeros(lwork, dtype=typ)
        u, sigma, bnd, v, info = lansvd_irl(which, jobu, jobv, m, n,
                                            shifts, k, maxiter, aprod,
                                            u, v, tol, work, cwork, iwork,
                                            dsoption, ioption, dparm, iparm)
    else:
        u, sigma, bnd, v, info = lansvd_irl(which, jobu, jobv, m, n,
                                            shifts, k, maxiter, aprod,
                                            u, v, tol, work, iwork,
                                            dsoption, ioption, dparm, iparm)

    # construct return tuple
    ret = ()
    if compute_u:
        ret = ret + (u[:, :k],)
    ret = ret + (sigma,)
    if compute_v:
        ret = ret + (v[:, :k].conj().T,)

    if full_output:
        ret = ret + (info, bnd)

    return ret
