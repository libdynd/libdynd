//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TYPE_ID_HPP_
#define _DYND__TYPE_ID_HPP_

#include <iostream>
#include <complex>

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
    // A dimension made up of offsets
    offset_dim_type_id,
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
    unary_expr_type_id,
    groupby_type_id,

    // Instances of this dtype are themselves dtypes
    dtype_type_id,

    // The number of built-in, atomic types (including uninitialized and void)
    builtin_type_id_count = 15
};

enum dtype_flags_t {
    // A symbolic name instead of just "0"
    dtype_flag_none = 0x00000000,
    // The dtype should be considered as a scalar
    dtype_flag_scalar = 0x00000001,
    // Memory of this dtype should be zero-initialized
    dtype_flag_zeroinit = 0x00000002,
    // Instances of this dtype point into other memory
    // blocks, e.g. string_dtype, var_dim_dtype.
    dtype_flag_blockref = 0x00000004,
    // Memory of this type must be destroyed,
    // e.g. it might hold a reference count or similar state
    dtype_flag_destructor = 0x00000008
};

enum {
    // These are the flags expression dtypes should inherit
    // from their operand type
    dtype_flags_operand_inherited =
                    dtype_flag_zeroinit|
                    dtype_flag_blockref|
                    dtype_flag_destructor,
    // These are the flags expression dtypes should inherit
    // from their value type
    dtype_flags_value_inherited =
                    dtype_flag_scalar
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

// A boolean class for dynamicndarray which is one-byte big
class dynd_bool {
    char m_value;
public:
    dynd_bool() : m_value(0) {}

    dynd_bool(bool value) : m_value(value) {}

    // Special case complex conversion to avoid ambiguous overload
    template<class T>
    dynd_bool(std::complex<T> value) : m_value(value != std::complex<T>(0)) {}

    operator bool() const {
        return m_value != 0;
    }
};


namespace detail {
    // Simple metaprogram taking log base 2 of 1, 2, 4, and 8
    template <int I> struct log2_x;
    template <> struct log2_x<1> {
        enum {value = 0};
    };
    template <> struct log2_x<2> {
        enum {value = 1};
    };
    template <> struct log2_x<4> {
        enum {value = 2};
    };
    template <> struct log2_x<8> {
        enum {value = 3};
    };
}


// Type trait for the type id
template <typename T> struct type_id_of;

template <typename T> struct type_id_of<const T> {enum {value = type_id_of<T>::value};};

// Can't use bool, because it doesn't have a guaranteed sizeof
template <> struct type_id_of<dynd_bool> {enum {value = bool_type_id};};
template <> struct type_id_of<char> {enum {value = ((char)-1) < 0 ? int8_type_id : uint8_type_id};};
template <> struct type_id_of<signed char> {enum {value = int8_type_id};};
template <> struct type_id_of<short> {enum {value = int16_type_id};};
template <> struct type_id_of<int> {enum {value = int32_type_id};};
template <> struct type_id_of<long> {
    enum {value = int8_type_id + detail::log2_x<sizeof(long)>::value};
};
template <> struct type_id_of<long long> {enum {value = int64_type_id};};
template <> struct type_id_of<uint8_t> {enum {value = uint8_type_id};};
template <> struct type_id_of<uint16_t> {enum {value = uint16_type_id};};
template <> struct type_id_of<unsigned int> {enum {value = uint32_type_id};};
template <> struct type_id_of<unsigned long> {
    enum {value = uint8_type_id + detail::log2_x<sizeof(unsigned long)>::value};
};
template <> struct type_id_of<unsigned long long>{enum {value = uint64_type_id};};
template <> struct type_id_of<float> {enum {value = float32_type_id};};
template <> struct type_id_of<double> {enum {value = float64_type_id};};
template <> struct type_id_of<std::complex<float> > {enum {value = complex_float32_type_id};};
template <> struct type_id_of<std::complex<double> > {enum {value = complex_float64_type_id};};
template <> struct type_id_of<void> {enum {value = void_type_id};};

// Type trait for the kind
template <typename T> struct dtype_kind_of;

template <> struct dtype_kind_of<void> {static const dtype_kind_t value = void_kind;};
// Can't use bool, because it doesn't have a guaranteed sizeof
template <> struct dtype_kind_of<dynd_bool> {static const dtype_kind_t value = bool_kind;};
template <> struct dtype_kind_of<char> {
    static const dtype_kind_t value = ((char)-1) < 0 ? int_kind : uint_kind;
};
template <> struct dtype_kind_of<signed char> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<short> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<int> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<long> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<long long> {static const dtype_kind_t value = int_kind;};
template <> struct dtype_kind_of<uint8_t> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<uint16_t> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<unsigned int> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<unsigned long> {static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<unsigned long long>{static const dtype_kind_t value = uint_kind;};
template <> struct dtype_kind_of<float> {static const dtype_kind_t value = real_kind;};
template <> struct dtype_kind_of<double> {static const dtype_kind_t value = real_kind;};
template <typename T> struct dtype_kind_of<std::complex<T> > {static const dtype_kind_t value = complex_kind;};

// Metaprogram for determining if a type is a valid C++ scalar
// of a particular dtype.
template<typename T> struct is_dtype_scalar {enum {value = false};};
template <> struct is_dtype_scalar<dynd_bool> {enum {value = true};};
template <> struct is_dtype_scalar<char> {enum {value = true};};
template <> struct is_dtype_scalar<signed char> {enum {value = true};};
template <> struct is_dtype_scalar<short> {enum {value = true};};
template <> struct is_dtype_scalar<int> {enum {value = true};};
template <> struct is_dtype_scalar<long> {enum {value = true};};
template <> struct is_dtype_scalar<long long> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned char> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned short> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned int> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned long> {enum {value = true};};
template <> struct is_dtype_scalar<unsigned long long> {enum {value = true};};
template <> struct is_dtype_scalar<float> {enum {value = true};};
template <> struct is_dtype_scalar<double> {enum {value = true};};
template <> struct is_dtype_scalar<std::complex<float> > {enum {value = true};};
template <> struct is_dtype_scalar<std::complex<double> > {enum {value = true};};

// Metaprogram for determining scalar alignment
template<typename T> struct scalar_align_of;
template <> struct scalar_align_of<dynd_bool> {enum {value = 1};};
template <> struct scalar_align_of<char> {enum {value = 1};};
template <> struct scalar_align_of<signed char> {enum {value = 1};};
template <> struct scalar_align_of<short> {enum {value = sizeof(short)};};
template <> struct scalar_align_of<int> {enum {value = sizeof(int)};};
template <> struct scalar_align_of<long> {enum {value = sizeof(long)};};
template <> struct scalar_align_of<long long> {enum {value = sizeof(long long)};};
template <> struct scalar_align_of<unsigned char> {enum {value = 1};};
template <> struct scalar_align_of<unsigned short> {enum {value = sizeof(unsigned short)};};
template <> struct scalar_align_of<unsigned int> {enum {value = sizeof(unsigned int)};};
template <> struct scalar_align_of<unsigned long> {enum {value = sizeof(unsigned long)};};
template <> struct scalar_align_of<unsigned long long> {enum {value = sizeof(unsigned long long)};};
template <> struct scalar_align_of<float> {enum {value = sizeof(long)};};
template <> struct scalar_align_of<double> {enum {value = sizeof(double)};};
template <> struct scalar_align_of<std::complex<float> > {enum {value = sizeof(long)};};
template <> struct scalar_align_of<std::complex<double> > {enum {value = sizeof(double)};};

// Metaprogram for determining if a type is the C++ "bool" or not
template<typename T> struct is_type_bool {enum {value = false};};
template<> struct is_type_bool<bool> {enum {value = true};};


} // namespace dynd

#endif // _DYND__TYPE_ID_HPP_
