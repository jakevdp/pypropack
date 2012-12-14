A Python Wrapper for PROPACK
============================
Author: Jake Vanderplas <vanderplas@astro.washington.edu>
License: BSD

**This is a work in progress: not all routines are currently wrapped, and
  the code should be considered unsupported**

About
-----
This is an experimental python wrapper for the PROPACK package, which
implements efficient singular value decompositions of large sparse matrices
and linear operators.  If the performance of this module is satisfactory, it
may be incorporated into Scipy at a later date.

Installation & Testing
----------------------
To install, type

    python setup.py build_ext --inplace

To run the few unit tests that currently exist, type

    nosetests pypropack/tests


Usage
-----
Currently very straightforward.  You can do

    >>> from pypropack import svdp
    >>> import numpy as np   
    >>> A = np.random.random((10, 20))
    >>> u, s, vt = svdp(A, k=3)  # compute top k singular values

The ``svdp`` documentation has more information.

Benchmarks
----------
use the script ``plot_benchmarks.py`` for a comparison of PROPACK, ARPACK,
SVDLIBC, and the standard LAPACK approaches to computing the svd.

PROPACK modifications
---------------------
PROPACK had to be slightly modified to play well with gfortran.
The PROPACK code used is version 2.1, downloaded from
http://soi.stanford.edu/~rmunk/PROPACK/

For compatibility with gfortran, the files ``PROPACK/*/second.f``
needed to be slightly modified.  The line

    REAL      ETIME
    EXTERNAL  ETIME

was changed to

    EXTERNAL REAL ETIME