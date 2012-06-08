# Cython notes:
#
# * Cython doesn't support 'const'. Hopefully this doesn't make it too
#   difficult to interact with const-correct code.
# * C++ 'bool' is spelled 'bint' in Cython.
# * Template functions are unsupported (classes: yes, functions: no)
# * Cython files may not contain UTF8
# * Overloading operator= is not supported
# * BUG: The "a(self.v)" idiom doesn't work with __add__, but works
#        with other functions.

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

cdef extern from "dnd/dtype.hpp" namespace "dnd":
    cdef cppclass shared_ptr[T]:
        T* get()

    cdef enum dtype_kind_t:
        bool_kind
        int_kind
        uint_kind
        real_kind
        complex_kind
        string_kind
        composite_kind
        expression_kind
        pattern_kind
        custom_kind

    cdef enum type_id_t:
        bool_type_id
        int8_type_id
        int16_type_id
        int32_type_id
        int64_type_id
        uint8_type_id
        uint16_type_id
        uint32_type_id
        uint64_type_id
        float32_type_id
        float64_type_id
        complex_float32_type_id
        complex_float64_type_id
        sse128f_type_id
        sse128d_type_id
        utf8_type_id
        struct_type_id
        tuple_type_id
        array_type_id
        conversion_type_id
        pattern_type_id

    cdef cppclass extended_dtype:
        type_id_t type_id()
        dtype_kind_t kind()
        unsigned char alignment()
        uintptr_t itemsize()
        dtype& value_dtype(dtype&)
        dtype& operand_dtype(dtype&)
        bint is_object_type()

    cdef cppclass dtype:
        dtype()
        dtype(type_id_t) except +
        dtype(type_id_t, uintptr_t) except +
        dtype(string&) except +
        bint operator==(dtype&)
        bint operator!=(dtype&)
        
        dtype& value_dtype()
        dtype& storage_dtype()
        type_id_t type_id()
        dtype_kind_t kind()
        int alignment()
        uintptr_t itemsize()
        bint is_object_type()
        extended_dtype* extended()

cdef extern from "dtype_functions.hpp" namespace "pydnd":
    string dtype_str(dtype&)
    string dtype_repr(dtype&)

cdef extern from "dnd/dtype_assign.hpp" namespace "dnd":
    cdef enum assign_error_mode:
        assign_error_none
        assign_error_overflow
        assign_error_fractional
        assign_error_inexact

cdef extern from "dnd/ndarray.hpp" namespace "dnd":
    cdef cppclass ndarray:
        ndarray() except +
        ndarray(signed char value)
        ndarray(short value)
        ndarray(int value)
        ndarray(long value)
        ndarray(long long value)
        ndarray(unsigned char value)
        ndarray(unsigned short value)
        ndarray(unsigned int value)
        ndarray(unsigned long value)
        ndarray(unsigned long long value)
        ndarray(float value)
        ndarray(double value)
        ndarray(complex[float] value)
        ndarray(complex[double] value)
        ndarray(dtype&)
        ndarray(dtype, int, intptr_t *, int *)

        dtype& get_dtype()
        int get_ndim()
        intptr_t* get_shape()
        intptr_t* get_strides()
        intptr_t get_num_elements()

        char* get_originptr()

        void val_assign(ndarray&, assign_error_mode)
        void val_assign(dtype&, char*, assign_error_mode)

        ndarray as_dtype(dtype&, assign_error_mode)

        void debug_dump(ostream&)

cdef extern from "ndarray_functions.hpp" namespace "pydnd":
    string ndarray_str(ndarray&)
    string ndarray_repr(ndarray&)
    string ndarray_debug_dump(ndarray&)

    void ndarray_init(ndarray&, object obj) except +
    void ndarray_init(ndarray&, object obj, dtype&) except +
    ndarray ndarray_vals(ndarray& n) except +

    ndarray ndarray_add(ndarray&, ndarray&) except +

cdef extern from "placement_wrappers.hpp" namespace "pydnd":
    cdef struct dtype_placement_wrapper:
        pass
    void dtype_placement_new(dtype_placement_wrapper&)
    void dtype_placement_new(dtype_placement_wrapper&, char*) except +
    void dtype_placement_delete(dtype_placement_wrapper&)
    # dtype placement cast
    dtype& a(dtype_placement_wrapper&)
    # dtype placement assignment
    void a(dtype_placement_wrapper&, dtype&)

    cdef struct ndarray_placement_wrapper:
        pass
    void ndarray_placement_new(ndarray_placement_wrapper&)
    void ndarray_placement_delete(ndarray_placement_wrapper&)
    # ndarray placement cast
    ndarray& a(ndarray_placement_wrapper&)
    # ndarray placement assignment
    void a(ndarray_placement_wrapper&, ndarray&)
