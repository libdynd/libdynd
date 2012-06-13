#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#
# Cython notes:
#
# * Cython doesn't support 'const'. Hopefully this doesn't make it too
#   difficult to interact with const-correct code.
# * C++ 'bool' is spelled 'bint' in Cython.
# * Template functions are unsupported (classes: yes, functions: no)
# * Cython files may not contain UTF8
# * Overloading operator= is not supported
# * BUG: The "GET(self.v)" idiom doesn't work with __add__, but works
#        with other functions. Requires a manual <w_...> cast to
#        make it work.
# * BUG: The "except +" annotation doesn't seem to work for overloaded
#        operators, exceptions weren't being caught.

include "libcpp/string.pxd"

cdef extern from "<stdint.h>":
    # From the Cython docs:
    #   If the header file uses typedef names such as word to refer
    #   to platform-dependent flavours of numeric types, you will
    #   need a corresponding ctypedef statement, but you don't need
    #   to match the type exactly, just use something of the right
    #   general kind (int, float, etc).
    ctypedef int intptr_t
    ctypedef unsigned int uintptr_t

cdef extern from "<complex>" namespace "std":
    cdef cppclass complex[T]:
        T real()
        T imag()

cdef extern from "<iostream>" namespace "std":
    cdef cppclass ostream:
        pass

    extern ostream cout


cdef extern from "placement_wrappers.hpp" namespace "pydnd":
    cdef struct dtype_placement_wrapper:
        pass
    void dtype_placement_new(dtype_placement_wrapper&)
    void dtype_placement_delete(dtype_placement_wrapper&)
    # dtype placement cast
    dtype& GET(dtype_placement_wrapper&)
    # dtype placement assignment
    void SET(dtype_placement_wrapper&, dtype&)

    cdef struct ndarray_placement_wrapper:
        pass
    void ndarray_placement_new(ndarray_placement_wrapper&)
    void ndarray_placement_delete(ndarray_placement_wrapper&)
    # ndarray placement cast
    ndarray& GET(ndarray_placement_wrapper&)
    # ndarray placement assignment
    void SET(ndarray_placement_wrapper&, ndarray&)
