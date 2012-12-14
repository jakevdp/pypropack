import numpy as np
import _spropack
import _dpropack


_type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z'}
_ndigits = {'f': 5, 'd': 12, 'F': 5, 'D': 12}
_lansvd_dict = {'f': _spropack.slansvd,
                'd': _dpropack.dlansvd}
_lansvd_irl_dict = {'f': _spropack.slansvd_irl,
                    'd': _dpropack.dlansvd_irl}


def svdp(A, k=3, kmax=None, compute_u=True, compute_v=True, tol=1E-5):
    """Compute the svd of A using PROPACK

    Parameters
    ----------
    A : array_like
        array of type float32 or float64
    k : int
        number of singular values/vectors to compute
    kmax : int
        maximal number of iterations / maximal dimension of Krylov subspace.
        default is 5 * k
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
    """
    A = np.asarray(A)
    typ = A.dtype.char

    lansvd = _lansvd_dict[typ]

    def aprod(transa, m, n, x, y, sparm, iparm):
        if transa.lower() == 'n':
            y[:m] = np.dot(A, x[:n])
        else:
            y[:n] = np.dot(A.T, x[:m])

    m, n = A.shape

    if compute_u:
        jobu = 'y'
    else:
        jobu = 'n'

    if compute_v:
        jobv = 'y'
    else:
        jobv = 'n'

    if kmax is None:
        kmax = 5 * k  # just a guess

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

    lwork = (m + n
             + 9 * kmax
             + 5 * kmax * kmax
             + max(3 * kmax **2 + 4 * kmax + 4,
                   3 * max(m, n)))
    work = np.zeros(lwork, dtype='f')

    liwork = 8 * kmax
    iwork = np.zeros(liwork, dtype='i')
    
    # dummy arguments: these are passed to aprod.
    dparm = np.zeros(1, dtype='f')
    iparm = np.zeros(1, dtype='i')

    res = _lansvd_dict[typ](jobu, jobv, m, n, k, aprod, u, v, tol,
                            work, iwork, doption, ioption, dparm, iparm)
    u, sigma, bnd, v, work, iwork, info = res

    return u[:, :k], sigma, v[:, :k].T
