#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

# Initialize Numpy
cdef extern from "do_import_array.hpp":
    pass
cdef extern from "numpy_interop.hpp" namespace "pydnd":
    object ndarray_as_numpy_struct_capsule(ndarray&) except +
    void import_numpy()
import_numpy()

# Initialize ctypes C level interop data
cdef extern from "ctypes_interop.hpp" namespace "pydnd":
    void init_ctypes_interop() except +
init_ctypes_interop()

# Initialize C++ access to the Cython type objects
init_w_ndarray_typeobject(w_ndarray)
init_w_dtype_typeobject(w_dtype)

# Issue a performance warning if any of the diagnostics macros are enabled
cdef extern from "<dnd/diagnostics.hpp>" namespace "dnd":
    bint any_diagnostics_enabled()
    string which_diagnostics_enabled()
if any_diagnostics_enabled():
    import warnings
    class PerformanceWarning(Warning):
        pass
    warnings.warn("Performance is reduced because of enabled diagnostics:\n" +
                str(which_diagnostics_enabled().c_str()), PerformanceWarning)

include "dnd.pxd"
include "dtype.pxd"
include "ndarray.pxd"
include "unary_gfunc.pxd"

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
            SET(self.v, make_dtype_from_object(rep))
    def __dealloc__(self):
        dtype_placement_delete(self.v)

    property element_size:
        def __get__(self):
            return GET(self.v).element_size()

    property alignment:
        def __get__(self):
            return GET(self.v).alignment()

    property string_encoding:
        def __get__(self):
            cdef string_encoding_t encoding = GET(self.v).string_encoding()
            if encoding == string_encoding_ascii:
                return "ascii"
            elif encoding == string_encoding_utf_8:
                return "utf_8"
            elif encoding == string_encoding_utf_16:
                return "utf_16"
            elif encoding == string_encoding_utf_32:
                return "utf_32"
            else:
                return "unknown_encoding"

    property value_dtype:
        """What this dtype looks like to calculations, printing, etc."""
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).value_dtype())
            return result

    property operand_dtype:
        """The next dtype down in the expression dtype chain."""
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).operand_dtype())
            return result

    property storage_dtype:
        """The bottom dtype in the expression chain."""
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).storage_dtype())
            return result

    def __str__(self):
        return str(dtype_str(GET(self.v)).c_str())

    def __repr__(self):
        return str(dtype_repr(GET(self.v)).c_str())

    def __richcmp__(lhs, rhs, int op):
        if op == Py_EQ:
            if type(lhs) == w_dtype and type(rhs) == w_dtype:
                return GET((<w_dtype>lhs).v) == GET((<w_dtype>rhs).v)
            else:
                return False
        elif op == Py_NE:
            if type(lhs) == w_dtype and type(rhs) == w_dtype:
                return GET((<w_dtype>lhs).v) != GET((<w_dtype>rhs).v)
            else:
                return False
        return NotImplemented

def make_byteswap_dtype(native_dtype, operand_dtype=None):
    """Constructs a byteswap dtype from a builtin one, with data feeding in from an optional operand dtype."""
    cdef w_dtype result = w_dtype()
    if operand_dtype is None:
        SET(result.v, dnd_make_byteswap_dtype(GET(w_dtype(native_dtype).v)))
    else:
        SET(result.v, dnd_make_byteswap_dtype(GET(w_dtype(native_dtype).v), GET(w_dtype(operand_dtype).v)))
    return result

def make_bytes_dtype(int element_size, int alignment):
    """Constructs a bytes dtype with the specified element size and alignment."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_bytes_dtype(element_size, alignment))
    return result

def make_convert_dtype(to_dtype, from_dtype):
    """Constructs a conversion dtype from the given source and destination dtypes."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_conversion_dtype(GET(w_dtype(to_dtype).v), GET(w_dtype(from_dtype).v), assign_error_fractional))
    return result

def make_unaligned_dtype(aligned_dtype):
    """Constructs a dtype with alignment of 1 from the given dtype."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_unaligned_dtype(GET(w_dtype(aligned_dtype).v)))
    return result

def make_fixedstring_dtype(encoding, int size):
    """Constructs a fixed-size string dtype with a specified encoding."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_fixedstring_dtype(encoding, size))
    return result

##############################################################################

# NOTE: This is a possible alternative to the init_w_ndarray_typeobject() call
#       above, but it generates a 1300 line header file and still requires calling
#       import__dnd from the C++ code, so directly using C++ primitives seems simpler.
#cdef public api class w_ndarray [object WNDArrayObject, type WNDArrayObject_Type]:

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
            ndarray_init_from_pyobject(GET(self.v), obj)

            # If a specific dtype is requested, use as_dtype to switch types
            if dtype is not None:
                SET(self.v, GET(self.v).as_dtype(GET(w_dtype(dtype).v), assign_error_fractional))
    def __dealloc__(self):
        ndarray_placement_delete(self.v)

    def debug_dump(self):
        """Prints a raw representation of the ndarray data."""
        print str(ndarray_debug_dump(GET(self.v)).c_str())

    def vals(self):
        """Returns a version of the ndarray with plain values, all expressions evaluated."""
        cdef w_ndarray result = w_ndarray()
        SET(result.v, ndarray_vals(GET(self.v)))
        return result

    def storage(self):
        """Returns a version of the ndarray with its storage dtype, all expressions discarded."""
        cdef w_ndarray result = w_ndarray()
        SET(result.v, GET(self.v).storage())
        return result

    def val_assign(self, obj):
        """Assigns to the ndarray by value instead of by reference."""
        cdef w_ndarray n = w_ndarray(obj)
        GET(self.v).val_assign(GET(n.v), assign_error_fractional)

    def as_dtype(self, dtype):
        """Converts the ndarray to the requested dtype. If dtype is an expression dtype, its expression gets applied on top of the existing data."""
        cdef w_ndarray result = w_ndarray()
        SET(result.v, GET(self.v).as_dtype(GET(w_dtype(dtype).v), assign_error_fractional))
        return result

    def view_as_dtype(self, dtype):
        """Views the data of the ndarray as the requested dtype, where it makes sense."""
        cdef w_ndarray result = w_ndarray()
        SET(result.v, GET(self.v).view_as_dtype(GET(w_dtype(dtype).v)))
        return result

    property dtype:
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).get_dtype())
            return result

    property ndim:
        def __get__(self):
            return GET(self.v).get_ndim()

    property shape:
        def __get__(self):
            return intptr_array_as_tuple(GET(self.v).get_ndim(), GET(self.v).get_shape())

    property strides:
        def __get__(self):
            return intptr_array_as_tuple(GET(self.v).get_ndim(), GET(self.v).get_strides())

    def __str__(self):
        return str(ndarray_str(GET(self.v)).c_str())

    def __repr__(self):
        return str(ndarray_repr(GET(self.v)).c_str())

    property __array_struct__:
        # Using the __array_struct__ mechanism to expose our data to numpy
        def __get__(self):
            return ndarray_as_numpy_struct_capsule(GET(self.v))

    def __add__(lhs, rhs):
        cdef w_ndarray result = w_ndarray()
        SET(result.v, ndarray_add(GET(w_ndarray(lhs).v), GET(w_ndarray(rhs).v)))
        return result

    def __sub__(lhs, rhs):
        cdef w_ndarray result = w_ndarray()
        SET(result.v, ndarray_subtract(GET(w_ndarray(lhs).v), GET(w_ndarray(rhs).v)))
        return result

    def __mul__(lhs, rhs):
        cdef w_ndarray result = w_ndarray()
        SET(result.v, ndarray_multiply(GET(w_ndarray(lhs).v), GET(w_ndarray(rhs).v)))
        return result

    def __div__(lhs, rhs):
        cdef w_ndarray result = w_ndarray()
        SET(result.v, ndarray_divide(GET(w_ndarray(lhs).v), GET(w_ndarray(rhs).v)))
        return result

cdef class w_unary_gfunc:
    cdef unary_gfunc_placement_wrapper v

    def __cinit__(self, bytes name):
        unary_gfunc_placement_new(self.v, name)
    def __dealloc__(self):
        unary_gfunc_placement_delete(self.v)

    property name:
        def __get__(self):
            return str(GET(self.v).get_name().c_str())

    def add_kernel(self, kernel):
        """Adds a kernel to the gfunc object. Currently, this means a ctypes object with prototype."""
        GET(self.v).add_kernel(kernel)

    def debug_dump(self):
        """Prints a raw representation of the gfunc data."""
        print str(GET(self.v).debug_dump().c_str())

    def __call__(self, *args, **kwargs):
        """Calls the gfunc."""
        GET(self.v).call(args, kwargs)
