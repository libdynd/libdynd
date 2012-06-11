cdef extern from "do_import_array.hpp":
    pass
cdef extern from "numpy_interop.hpp" namespace "pydnd":
    object ndarray_as_numpy_struct_capsule(ndarray&) except +
    void import_numpy()
import_numpy()

include "dnd.pxd"
include "dtype.pxd"
include "ndarray.pxd"

from cython.operator import dereference

cdef class w_dtype:
    # To access the embedded dtype, use "a(self.v)",
    # which returns a reference to the dtype.
    cdef dtype_placement_wrapper v

    def __cinit__(self, rep=None):
        dtype_placement_new(self.v)
        if rep is not None:
            if type(rep) is w_dtype:
                a(self.v, a((<w_dtype>rep).v))
            else:
                a(self.v, make_dtype_from_object(rep))
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

    def __cinit__(self, obj=None, dtype=None):
        ndarray_placement_new(self.v)
        if obj is not None:
            # Get the array data
            if type(obj) is w_ndarray:
                a(self.v, a((<w_ndarray>obj).v))
            else:
                ndarray_init(a(self.v), obj)

            # If a specific dtype is requested, use as_dtype to switch types
            if dtype is not None:
                a(self.v, a(self.v).as_dtype(a(w_dtype(dtype).v), assign_error_fractional))
    def __dealloc__(self):
        ndarray_placement_delete(self.v)

    def debug_dump(self):
        """Prints a raw representation of the ndarray data."""
        print str(ndarray_debug_dump(a(self.v)).c_str())

    def vals(self):
        """Returns a version of the ndarray with plain values, no expressions."""
        cdef w_ndarray result
        result = w_ndarray()
        a(result.v, ndarray_vals(a(self.v)))
        return result

    def val_assign(self, obj):
        """Assigns to the ndarray by value instead of by reference."""
        cdef w_ndarray n
        n = w_ndarray(obj)
        a(self.v).val_assign(a(n.v), assign_error_fractional)

    property dtype:
        def __get__(self):
            cdef w_dtype result
            result = w_dtype()
            a(result.v, a(self.v).get_dtype())
            return result

    def __str__(self):
        return str(ndarray_str(a(self.v)).c_str())

    def __repr__(self):
        return str(ndarray_repr(a(self.v)).c_str())

    property __array_struct__:
        # Using the __array_struct__ mechanism to expose our data to numpy
        def __get__(self):
            return ndarray_as_numpy_struct_capsule(a(self.v))

    def __add__(self, rhs):
        cdef w_ndarray result = w_ndarray()
        # Cython seems to lose the w_ndarray type information about "self", need to forcefully cast :P
        a(result.v, a((<w_ndarray>self).v) + a(w_ndarray(rhs).v))
        return result

    def __radd__(self, lhs):
        # TODO: __r<*>__ are crashing, seems to be a Cython bug. Need to investigate...
        cdef w_ndarray result = w_ndarray()
        a(result.v, a(w_ndarray(lhs).v) + a((<w_ndarray>self).v))
        return result

    def __sub__(self, rhs):
        cdef w_ndarray result = w_ndarray()
        a(result.v, a((<w_ndarray>self).v) - a(w_ndarray(rhs).v))
        return result

    def __rsub__(self, lhs):
        cdef w_ndarray result = w_ndarray()
        a(result.v, a(w_ndarray(lhs).v) - a((<w_ndarray>self).v))
        return result

    def __mul__(self, rhs):
        cdef w_ndarray result = w_ndarray()
        a(result.v, a((<w_ndarray>self).v) * a(w_ndarray(rhs).v))
        return result

    def __rmul__(self, lhs):
        cdef w_ndarray result = w_ndarray()
        a(result.v, a(w_ndarray(lhs).v) * a((<w_ndarray>self).v))
        return result

    def __div__(self, rhs):
        cdef w_ndarray result = w_ndarray()
        a(result.v, a((<w_ndarray>self).v) / a(w_ndarray(rhs).v))
        return result

    def __rdiv__(self, lhs):
        cdef w_ndarray result = w_ndarray()
        a(result.v, a(w_ndarray(lhs).v) / a((<w_ndarray>self).v))
        return result
