//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TYPE_ID_HPP_
#define _DYND__TYPE_ID_HPP_

#include <iostream>

#include <dynd/config.hpp>

namespace dynd {

enum dtype_kind_t {
    bool_kind,
    int_kind,
    uint_kind,
    real_kind,
    complex_kind,
    // string_kind means subclass of base_string_dtype
    string_kind,
    bytes_kind,
    void_kind,
    datetime_kind,
    // For any dimension dtypes which have elements of all the same type
    uniform_dim_kind,
    // For struct_type_id and fixedstruct_type_id
    struct_kind,
    // For dtypes whose value_dtype != the dtype, signals
    // that calculations should look at the value_dtype for
    // type promotion, etc.
    expression_kind,
    // For pattern-matching dtypes
    pattern_kind,
    // For use when it becomes possible to register custom dtypes
    custom_kind
};

enum type_id_t {
    // The value zero is reserved for an uninitialized dtype.
    uninitialized_type_id,
    // A 1-byte boolean type
    bool_type_id,
    // Signed integer types
    int8_type_id,
    int16_type_id,
    int32_type_id,
    int64_type_id,
    // Unsigned integer types
    uint8_type_id,
    uint16_type_id,
    uint32_type_id,
    uint64_type_id,
    // Floating point types
    float32_type_id,
    float64_type_id,
    // Complex floating-point types
    complex_float32_type_id,
    complex_float64_type_id,
    // Means no type, just like in C. (Different from NumPy)
    void_type_id,
    // Like C/C++ (void*), the storage of pointer_dtype
    void_pointer_type_id,

    // A pointer type
    pointer_type_id,

    // blockref primitive dtypes
    bytes_type_id,
    // A bytes buffer of a fixed size
    fixedbytes_type_id,

    // A variable-sized string type
    string_type_id,
    // A NULL-terminated string buffer of a fixed size
    fixedstring_type_id,

    // A categorical (enum-like) type
    categorical_type_id,
    // A 32-bit date type
    date_type_id,
    // A 32-bit date type limited to business days
    busdate_type_id,
    // A UTF-8 encoded string type for holding JSON
    json_type_id,

    // A strided array dimension type (like NumPy)
    strided_dim_type_id,
    // A fixed-sized array dimension type
    fixed_dim_type_id,
    // A variable-sized array dimension type
    var_dim_type_id,

    // A struct type with variable layout
    struct_type_id,
    // A struct type with fixed layout
    fixedstruct_type_id,
    tuple_type_id,
    ndobject_type_id,

    // Adapter dtypes
    convert_type_id,
    byteswap_type_id,
    view_type_id,

    // A type for property access
    property_type_id,

    // Advanced expression dtypes
    expr_type_id,
    groupby_type_id,

    // The number of built-in, atomic types (including uninitialized and void)
    builtin_type_id_count = 15
};

enum dtype_flags_t {
    // A symbolic name instead of just "0"
    dtype_flag_none = 0x00000000,
    // The dtype should be considered as a scalar
    dtype_flag_scalar = 0x00000001,
    // Memory of this dtype should be zero-initialized
    dtype_flag_zeroinit = 0x00000002
};

enum kernel_request_t {
    /** Kernel function unary_single_operation_t */
    kernel_request_single,
    /** Kernel function unary_strided_operation_t */
    kernel_request_strided,
    /**
     * Kernel function unary_single_operation_t,
     * but the data in the kernel at position 'offset_out'
     * is for data that describes the accumulation
     * of multiple strided dimensions that work
     * in a simple NumPy fashion.
     */
//    kernel_request_single_multistride,
    /**
     * Kernel function unary_strided_operation_t,
     * but the data in the kernel at position 'offset_out'
     * is for data that describes the accumulation
     * of multiple strided dimensions that work
     * in a simple NumPy fashion.
     */
//    kernel_request_strided_multistride
};

std::ostream& operator<<(std::ostream& o, dtype_kind_t kind);
std::ostream& operator<<(std::ostream& o, type_id_t tid);
std::ostream& operator<<(std::ostream& o, kernel_request_t kernreq);

enum {
    /** A mask within which alll the built-in type ids are guaranteed to fit */
    builtin_type_id_mask = 0x1f
};

// Forward declaration so we can make the is_builtin_dtype function here
class base_dtype;

inline bool is_builtin_dtype(const base_dtype *dt) {
    return (reinterpret_cast<uintptr_t>(dt)&(~static_cast<uintptr_t>(builtin_type_id_mask))) == 0;
}

} // namespace dynd

#endif // _DYND__TYPE_ID_HPP_
