import numpy as np
import _spropack
import _dpropack

from scipy.sparse.linalg import aslinearoperator


_type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z'}
_ndigits = {'f': 5, 'd': 12, 'F': 5, 'D': 12}
_lansvd_dict = {'f': _spropack.slansvd,
                'd': _dpropack.dlansvd}
_lansvd_irl_dict = {'f': _spropack.slansvd_irl,
                    'd': _dpropack.dlansvd_irl}


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
        if transa.lower() == 'n':
            y[:m] = self.A.matvec(x[:n])
        else:
            y[:n] = self.A.rmatvec(x[:m])
    
    @property
    def shape(self):
        return self.A.shape

    @property
    def dtype(self):
        try:
            return self.A.dtype
        except:
            return A.matvec(np.zeros(A.shape[1])).dtype


def svdp(A, k=3, kmax=None, compute_u=True, compute_v=True, tol=1E-5):
    """Compute the svd of A using PROPACK

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
    tol : float
        desired relative accuracy for computed singular values

    Returns
    -------
    u : ndarray
        the top k left singular vectors, shape = (A.shape[0], 3)
    sigma : ndarray
        the top k singular values, shape = (k,)
    vt : ndarray
        the top k right singular vectors, shape = (3, A.shape[1])

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

    # options for the fit
    # TODO: explain these and make these parameters adjustable
    ioption = np.zeros(2, dtype='i')
    ioption[0] = 0  # controls Gram-Schmidt procedure
    ioption[1] = 1  # controls re-orthonormalization

    eps = 2 * np.finfo(typ).eps
    doption = np.zeros(3, dtype='f')
    doption[0] = eps ** 0.5   # level of orthogonality to maintain
    doption[1] = eps ** 0.75  # purge vectors larger than this
    doption[2] = 0.0          # estimate of ||A||

    # TODO: choose lwork based on inputs (see propack documentation)
    lwork = (m + n
             + 9 * kmax
             + 5 * kmax * kmax
             + max(3 * kmax **2 + 4 * kmax + 4,
                   3 * max(m, n)))
    work = np.zeros(lwork, dtype='f')

    # TODO: choose liwork based on inputs (see propack documentation)
    liwork = 8 * kmax
    iwork = np.zeros(liwork, dtype='i')
    
    # dummy arguments: these are passed to aprod, and not used
    dparm = np.zeros(1, dtype='f')
    iparm = np.zeros(1, dtype='i')

    u, sigma, bnd, v, work, iwork, info =\
        _lansvd_dict[typ](jobu, jobv, m, n, k, aprod, u, v, tol,
                          work, iwork, doption, ioption, dparm, iparm)

    # construct return tuple
    ret = ()
    if compute_u:
        ret = ret + (u[:, :k],)
    ret = ret + (sigma,)
    if compute_v:
        ret = ret + (v[:, :k].T,)

    return ret
