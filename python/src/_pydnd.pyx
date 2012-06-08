cdef extern from "do_import_array.hpp":
    pass

include "dnd.pxd"

cdef extern from "numpy/ndarrayobject.h":
    void import_array()
cdef extern from "numpy/ufuncobject.h":
    void import_umath()

import_array()
import_umath()

from cython.operator import dereference

cdef class w_dtype:
    # To access the embedded dtype, use "a(self.v)",
    # which returns a reference to the dtype.
    cdef dtype_placement_wrapper v

    def __cinit__(self, char* rep):
        dtype_placement_new(self.v, rep)
    def __dealloc__(self):
        dtype_placement_delete(self.v)

    def __str__(self):
        return str(dtype_str(a(self.v)).c_str())

    def __repr__(self):
        return str(dtype_repr(a(self.v)).c_str())

cdef class w_ndarray:
    # To access the embedded dtype, use "a(self.v)",
    # which returns a reference to the ndarray
    cdef ndarray_placement_wrapper v

    def __cinit__(self, obj=None, w_dtype dtype=None):
        ndarray_placement_new(self.v)
        if obj is not None:
            if dtype is None:
                ndarray_init(a(self.v), obj)
            else:
                ndarray_init(a(self.v), obj, a(dtype.v))
    def __dealloc__(self):
        ndarray_placement_delete(self.v)

    def debug_dump(self):
        """Prints a raw representation of the ndarray data."""
        print str(ndarray_debug_dump(a(self.v)).c_str())

    def __str__(self):
        return str(ndarray_str(a(self.v)).c_str())

    def __repr__(self):
        return str(ndarray_repr(a(self.v)).c_str())

    def __add__(self, other):
        cdef ndarray lhs, rhs
        # Cython seems to lose the w_ndarray type information about "self", need to forcefully cast :P
        lhs = a((<w_ndarray>self).v)
        rhs = a(w_ndarray(other).v)
        cdef w_ndarray result = w_ndarray()
        a(result.v, ndarray_add(lhs, rhs))
        return result

    def vals(self):
        """Returns a version of the ndarray with plain values, no expressions."""
        cdef w_ndarray result
        result = w_ndarray()
        a(result.v, ndarray_vals(a(self.v)))
        return result
