#!/usr/bin/env python
from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.system_info import get_info, NotFoundError
    from numpy.distutils.misc_util import Configuration

    config = Configuration('pypropack', parent_package, top_path)

    lapack_opt = get_info('lapack_opt')

    if not lapack_opt:
        raise NotFoundError('no lapack/blas resources found')

    #------------------------------------------------------------
    # Set up the libraries.
    #  We need a different python .so file for each, because they
    #  re-use names for functions in the LAPACK extensions.  This
    #  could probably be remedied with some work.
    type_dict = dict(s='single',
                     d='double',
                     c='complex8',
                     z='complex16')

    for typ in type_dict:
        lapack_lib = 'sp_%slapack_util' % typ
        propack_lib = '_%spropack' % typ

        lapack_sources = join('PROPACK', type_dict[typ], 'Lapack_Util', '*.f')
        propack_sources = join('PROPACK', type_dict[typ], '*.F')

        config.add_library(lapack_lib,
                           sources = lapack_sources)
        config.add_library(propack_lib,
                           sources = propack_sources)
        config.add_extension('_%spropack' % typ,
                             sources='%spropack.pyf' % typ,
                             libraries=[lapack_lib, propack_lib],
                             extra_info = lapack_opt)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
