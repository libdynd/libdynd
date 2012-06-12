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
    # To access the embedded dtype, use "GET(self.v)",
    # which returns a reference to the dtype, and
    # SET(self.v, <dtype value>), which sets the embeded
    # dtype's value.
    cdef dtype_placement_wrapper v

    def __cinit__(self, rep=None):
        dtype_placement_new(self.v)
        if rep is not None:
            if type(rep) is w_dtype:
                SET(self.v, GET((<w_dtype>rep).v))
            else:
                SET(self.v, make_dtype_from_object(rep))
    def __dealloc__(self):
        dtype_placement_delete(self.v)

    def __str__(self):
        return str(dtype_str(GET(self.v)).c_str())

    def __repr__(self):
        return str(dtype_repr(GET(self.v)).c_str())

cdef class w_ndarray:
    # To access the embedded dtype, use "GET(self.v)",
    # which returns a reference to the ndarray, and
    # SET(self.v, <ndarray value>), which sets the embeded
    # ndarray's value.
    cdef ndarray_placement_wrapper v

    def __cinit__(self, obj=None, dtype=None):
        ndarray_placement_new(self.v)
        if obj is not None:
            # Get the array data
            if type(obj) is w_ndarray:
                SET(self.v, GET((<w_ndarray>obj).v))
            else:
                ndarray_init(GET(self.v), obj)

            # If a specific dtype is requested, use as_dtype to switch types
            if dtype is not None:
                SET(self.v, GET(self.v).as_dtype(GET(w_dtype(dtype).v), assign_error_fractional))
    def __dealloc__(self):
        ndarray_placement_delete(self.v)

    def debug_dump(self):
        """Prints a raw representation of the ndarray data."""
        print str(ndarray_debug_dump(GET(self.v)).c_str())

    def vals(self):
        """Returns a version of the ndarray with plain values, no expressions."""
        cdef w_ndarray result
        result = w_ndarray()
        SET(result.v, ndarray_vals(GET(self.v)))
        return result

    def val_assign(self, obj):
        """Assigns to the ndarray by value instead of by reference."""
        cdef w_ndarray n
        n = w_ndarray(obj)
        GET(self.v).val_assign(GET(n.v), assign_error_fractional)

    property dtype:
        def __get__(self):
            cdef w_dtype result
            result = w_dtype()
            SET(result.v, GET(self.v).get_dtype())
            return result

    def __str__(self):
        return str(ndarray_str(GET(self.v)).c_str())

    def __repr__(self):
        return str(ndarray_repr(GET(self.v)).c_str())

    property __array_struct__:
        # Using the __array_struct__ mechanism to expose our data to numpy
        def __get__(self):
            return ndarray_as_numpy_struct_capsule(GET(self.v))

    def __add__(self, rhs):
        cdef w_ndarray result = w_ndarray()
        # Cython seems to lose the w_ndarray type information about "self", need to forcefully cast :P
        SET(result.v, ndarray_add(GET((<w_ndarray>self).v), GET(w_ndarray(rhs).v)))
        return result

    def __radd__(self, lhs):
        # TODO: __r<*>__ are crashing, seems to be a Cython bug. Need to investigate...
        cdef w_ndarray result = w_ndarray()
        SET(result.v, ndarray_add(GET(w_ndarray(lhs).v), GET((<w_ndarray>self).v)))
        return result

    def __sub__(self, rhs):
        cdef w_ndarray result = w_ndarray()
        SET(result.v, GET((<w_ndarray>self).v) - GET(w_ndarray(rhs).v))
        return result

    def __rsub__(self, lhs):
        cdef w_ndarray result = w_ndarray()
        SET(result.v, GET(w_ndarray(lhs).v) - GET((<w_ndarray>self).v))
        return result

    def __mul__(self, rhs):
        cdef w_ndarray result = w_ndarray()
        SET(result.v, GET((<w_ndarray>self).v) * GET(w_ndarray(rhs).v))
        return result

    def __rmul__(self, lhs):
        cdef w_ndarray result = w_ndarray()
        SET(result.v, GET(w_ndarray(lhs).v) * GET((<w_ndarray>self).v))
        return result

    def __div__(self, rhs):
        cdef w_ndarray result = w_ndarray()
        SET(result.v, GET((<w_ndarray>self).v) / GET(w_ndarray(rhs).v))
        return result

    def __rdiv__(self, lhs):
        cdef w_ndarray result = w_ndarray()
        SET(result.v, GET(w_ndarray(lhs).v) / GET((<w_ndarray>self).v))
        return result
