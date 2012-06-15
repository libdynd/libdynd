#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#
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
        dtype& operand_dtype()
        dtype& storage_dtype()
        type_id_t type_id()
        dtype_kind_t kind()
        int alignment()
        uintptr_t itemsize()
        bint is_object_type()
        extended_dtype* extended()

    dtype dnd_make_bytes_dtype "dnd::make_bytes_dtype" (intptr_t, intptr_t) except +

cdef extern from "dnd/dtype_assign.hpp" namespace "dnd":
    cdef enum assign_error_mode:
        assign_error_none
        assign_error_overflow
        assign_error_fractional
        assign_error_inexact

cdef extern from "dnd/dtypes/byteswap_dtype.hpp" namespace "dnd":
    dtype dnd_make_byteswap_dtype "dnd::make_byteswap_dtype" (dtype&) except +
    dtype dnd_make_byteswap_dtype "dnd::make_byteswap_dtype" (dtype&, dtype&) except +

cdef extern from "dnd/dtypes/conversion_dtype.hpp" namespace "dnd":
    dtype dnd_make_conversion_dtype "dnd::make_conversion_dtype" (dtype&, dtype&, assign_error_mode) except +

cdef extern from "dnd/dtypes/dtype_alignment.hpp" namespace "dnd":
    dtype dnd_make_unaligned_dtype "dnd::make_unaligned_dtype" (dtype&) except +

cdef extern from "dtype_functions.hpp" namespace "pydnd":
    string dtype_str(dtype&)
    string dtype_repr(dtype&)
    dtype deduce_dtype_from_object(object) except +
    dtype make_dtype_from_object(object) except +

