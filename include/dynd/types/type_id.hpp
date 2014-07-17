//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TYPE_ID_HPP_
#define _DYND__TYPE_ID_HPP_

#include <iostream>
#include <complex>

#include <dynd/config.hpp>
#include <dynd/types/dynd_int128.hpp>
#include <dynd/types/dynd_uint128.hpp>
#include <dynd/types/dynd_float16.hpp>
#include <dynd/types/dynd_float128.hpp>
#include <dynd/types/dynd_complex.hpp>

namespace dynd {

enum type_kind_t {
    bool_kind,
    int_kind,
    uint_kind,
    real_kind,
    complex_kind,
    char_kind,
    // string_kind means subclass of base_string_type
    string_kind,
    bytes_kind,
    void_kind,
    datetime_kind,
    // For any dimension types which have elements of all the same type
    dim_kind,
    // For struct_type_id and cstruct_type_id
    struct_kind,
    // For tuple_type_id and ctuple_type_id
    tuple_kind,
    // For types whose value itself is dynamically typed
    dynamic_kind,
    // For types whose value_type != the type, signals
    // that calculations should look at the value_type for
    // type promotion, etc.
    expr_kind,
    // For the option type, whose value may or may not be present
    option_kind,
    // For types that specify a memory space
    memory_kind,
    // For types containing type vars, or function prototypes that can't be
    // instantiated
    symbolic_kind,
    // For use when it becomes possible to register custom types
    custom_kind
};

enum type_id_t {
    // The value zero is reserved for an uninitialized type.
    uninitialized_type_id,
    // A 1-byte boolean type
    bool_type_id,
    // Signed integer types
    int8_type_id,
    int16_type_id,
    int32_type_id,
    int64_type_id,
    int128_type_id,
    // Unsigned integer types
    uint8_type_id,
    uint16_type_id,
    uint32_type_id,
    uint64_type_id,
    uint128_type_id,
    // Floating point types
    float16_type_id,
    float32_type_id,
    float64_type_id,
    float128_type_id,
    // Complex floating-point types
    complex_float32_type_id,
    complex_float64_type_id,
    // Means no type, just like in C. (Different from NumPy)
    void_type_id,
    // Like C/C++ (void*), the storage of pointer_type
    void_pointer_type_id,

    // A pointer type
    pointer_type_id,

    // blockref primitive types
    bytes_type_id,
    // A bytes buffer of a fixed size
    fixedbytes_type_id,

    // A single string character
    char_type_id,

    // A variable-sized string type
    string_type_id,
    // A NULL-terminated string buffer of a fixed size
    fixedstring_type_id,

    // A categorical (enum-like) type
    categorical_type_id,
    // A 32-bit date type
    date_type_id,
    // A 64-bit time type
    time_type_id,
    // A 64-bit datetime type
    datetime_type_id,
    // A 32-bit date type limited to business days
    busdate_type_id,
    // A UTF-8 encoded string type for holding JSON
    json_type_id,

    // A strided array dimension type (like NumPy)
    strided_dim_type_id,
    // A fixed-sized array dimension type
    fixed_dim_type_id,
    // A fixed-sized, fixed-stride array dimension type
    cfixed_dim_type_id,
    // A dimension made up of offsets
    offset_dim_type_id,
    // A variable-sized array dimension type
    var_dim_type_id,

    // A struct type with variable layout
    struct_type_id,
    // A struct type with fixed layout
    cstruct_type_id,
    // A tuple type with variable layout
    tuple_type_id,
    // A tuple type with fixed layout
    ctuple_type_id,

    option_type_id,

    ndarrayarg_type_id,

    // Adapter types
    adapt_type_id,
    convert_type_id,
    byteswap_type_id,
    view_type_id,

    // A CUDA host memory type
    cuda_host_type_id,
    // A CUDA device (global) memory type
    cuda_device_type_id,

    // A type for property access
    property_type_id,

    // Advanced expression types
    expr_type_id,
    unary_expr_type_id,
    groupby_type_id,

    // Instances of this type are themselves types
    type_type_id,

    // Instances of this type are arrfunc objects
    arrfunc_type_id,

    // Symbolic types
    funcproto_type_id,
    typevar_type_id,
    typevar_dim_type_id,
    ellipsis_dim_type_id,
    // A special type which holds a fragment of canonical dimensions
    // for the purpose of broadcasting together named ellipsis type vars.
    dim_fragment_type_id,

    // The number of built-in, atomic types (including uninitialized and void)
    builtin_type_id_count = 19
};
#define DYND_BUILTIN_TYPE_ID_COUNT 19

enum type_flags_t {
    // A symbolic name instead of just "0"
    type_flag_none = 0x00000000,
    // The type should be considered as a scalar
    type_flag_scalar = 0x00000001,
    // Memory of this type must be zero-initialized
    type_flag_zeroinit = 0x00000002,
    // Memory of this type must be constructed
    //type_flag_constructor = 0x00000004,
    // Instances of this type point into other memory
    // blocks, e.g. string_type, var_dim_type.
    type_flag_blockref = 0x00000008,
    // Memory of this type must be destroyed,
    // e.g. it might hold a reference count or similar state
    type_flag_destructor = 0x00000010,
    // Memory of this type is not readable directly from the host
    type_flag_not_host_readable = 0x00000020,
    // This type contains a symbolic construct like a type var
    type_flag_symbolic = 0x00000040,
};

enum axis_order_classification_t {
    // No order (0D, 1D arrays, higher-D arrays with at most one
    // dimension size greater than 1)
    axis_order_none,
    // Includes striding that goes "big small big" or "small big small",
    // so not compatible with C or F order
    axis_order_neither,
    // Includes at least one "small big" stride occurrence,
    // and no "big small"
    axis_order_f,
    // Includes at least one "big small" stride occurrence,
    // and no "small big"
    axis_order_c
};

enum {
    // These are the flags expression types should inherit
    // from their operand type
    type_flags_operand_inherited =
                    type_flag_zeroinit |
                    type_flag_blockref |
                    type_flag_destructor |
                    type_flag_not_host_readable |
                    type_flag_symbolic,
    // These are the flags expression types should inherit
    // from their value type
    type_flags_value_inherited =
                    type_flag_scalar |
                    type_flag_symbolic
};

std::ostream& operator<<(std::ostream& o, type_kind_t kind);
std::ostream& operator<<(std::ostream& o, type_id_t tid);

enum {
    /** A mask within which all the built-in type ids are guaranteed to fit */
    builtin_type_id_mask = 0x3f
};

// Forward declaration so we can make the is_builtin_type function here
class base_type;

inline bool is_builtin_type(const base_type *dt) {
    return (reinterpret_cast<uintptr_t>(dt)&(~static_cast<uintptr_t>(builtin_type_id_mask))) == 0;
}

// A boolean class for dynamicndarray which is one-byte big
class dynd_bool {
    char m_value;
public:
    DYND_CUDA_HOST_DEVICE dynd_bool() : m_value(0) {}

    DYND_CUDA_HOST_DEVICE dynd_bool(bool value) : m_value(value) {}

    // Special case complex conversion to avoid ambiguous overload
    template<class T>
    DYND_CUDA_HOST_DEVICE dynd_bool(dynd_complex<T> value) : m_value(value != dynd_complex<T>(0)) {}

    DYND_CUDA_HOST_DEVICE operator bool() const {
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
template <> struct type_id_of<dynd_int128> {enum {value = int128_type_id};};
template <> struct type_id_of<uint8_t> {enum {value = uint8_type_id};};
template <> struct type_id_of<uint16_t> {enum {value = uint16_type_id};};
template <> struct type_id_of<unsigned int> {enum {value = uint32_type_id};};
template <> struct type_id_of<unsigned long> {
    enum {value = uint8_type_id + detail::log2_x<sizeof(unsigned long)>::value};
};
template <> struct type_id_of<unsigned long long>{enum {value = uint64_type_id};};
template <> struct type_id_of<dynd_uint128> {enum {value = uint128_type_id};};
template <> struct type_id_of<dynd_float16> {enum {value = float16_type_id};};
template <> struct type_id_of<float> {enum {value = float32_type_id};};
template <> struct type_id_of<double> {enum {value = float64_type_id};};
template <> struct type_id_of<dynd_float128> {enum {value = float128_type_id};};
template <> struct type_id_of<dynd_complex<float> > {enum {value = complex_float32_type_id};};
template <> struct type_id_of<dynd_complex<double> > {enum {value = complex_float64_type_id};};
template <> struct type_id_of<void> {enum {value = void_type_id};};
// Also allow type_id_of<std::complex<>> as synonyms for type_id_of<dynd_complex<>>
template <> struct type_id_of<std::complex<float> > {enum {value = complex_float32_type_id};};
template <> struct type_id_of<std::complex<double> > {enum {value = complex_float64_type_id};};

// Type trait for the kind
template <typename T> struct dynd_kind_of;

template <> struct dynd_kind_of<void> {static const type_kind_t value = void_kind;};
// Can't use bool, because it doesn't have a guaranteed sizeof
template <> struct dynd_kind_of<dynd_bool> {static const type_kind_t value = bool_kind;};
template <> struct dynd_kind_of<char> {
    static const type_kind_t value = ((char)-1) < 0 ? int_kind : uint_kind;
};
template <> struct dynd_kind_of<signed char> {static const type_kind_t value = int_kind;};
template <> struct dynd_kind_of<short> {static const type_kind_t value = int_kind;};
template <> struct dynd_kind_of<int> {static const type_kind_t value = int_kind;};
template <> struct dynd_kind_of<long> {static const type_kind_t value = int_kind;};
template <> struct dynd_kind_of<long long> {static const type_kind_t value = int_kind;};
template <> struct dynd_kind_of<dynd_int128> {static const type_kind_t value = int_kind;};
template <> struct dynd_kind_of<uint8_t> {static const type_kind_t value = uint_kind;};
template <> struct dynd_kind_of<uint16_t> {static const type_kind_t value = uint_kind;};
template <> struct dynd_kind_of<unsigned int> {static const type_kind_t value = uint_kind;};
template <> struct dynd_kind_of<unsigned long> {static const type_kind_t value = uint_kind;};
template <> struct dynd_kind_of<unsigned long long>{static const type_kind_t value = uint_kind;};
template <> struct dynd_kind_of<dynd_uint128> {static const type_kind_t value = uint_kind;};
template <> struct dynd_kind_of<dynd_float16> {static const type_kind_t value = real_kind;};
template <> struct dynd_kind_of<float> {static const type_kind_t value = real_kind;};
template <> struct dynd_kind_of<double> {static const type_kind_t value = real_kind;};
template <> struct dynd_kind_of<dynd_float128> {static const type_kind_t value = real_kind;};
template <typename T> struct dynd_kind_of<dynd_complex<T> > {static const type_kind_t value = complex_kind;};

// Metaprogram for determining if a type is a valid C++ scalar
// of a particular type.
template<typename T> struct is_dynd_scalar {enum {value = false};};
template <> struct is_dynd_scalar<dynd_bool> {enum {value = true};};
template <> struct is_dynd_scalar<char> {enum {value = true};};
template <> struct is_dynd_scalar<signed char> {enum {value = true};};
template <> struct is_dynd_scalar<short> {enum {value = true};};
template <> struct is_dynd_scalar<int> {enum {value = true};};
template <> struct is_dynd_scalar<long> {enum {value = true};};
template <> struct is_dynd_scalar<long long> {enum {value = true};};
template <> struct is_dynd_scalar<dynd_int128> {enum {value = true};};
template <> struct is_dynd_scalar<unsigned char> {enum {value = true};};
template <> struct is_dynd_scalar<unsigned short> {enum {value = true};};
template <> struct is_dynd_scalar<unsigned int> {enum {value = true};};
template <> struct is_dynd_scalar<unsigned long> {enum {value = true};};
template <> struct is_dynd_scalar<unsigned long long> {enum {value = true};};
template <> struct is_dynd_scalar<dynd_uint128> {enum {value = true};};
template <> struct is_dynd_scalar<dynd_float16> {enum {value = true};};
template <> struct is_dynd_scalar<float> {enum {value = true};};
template <> struct is_dynd_scalar<double> {enum {value = true};};
template <> struct is_dynd_scalar<dynd_float128> {enum {value = true};};
template <> struct is_dynd_scalar<dynd_complex<float> > {enum {value = true};};
template <> struct is_dynd_scalar<dynd_complex<double> > {enum {value = true};};
// Allow std::complex as scalars equivalent to dynd_complex
template <> struct is_dynd_scalar<std::complex<float> > {enum {value = true};};
template <> struct is_dynd_scalar<std::complex<double> > {enum {value = true};};

// Metaprogram for determining scalar alignment
template <typename T> struct scalar_align_of {
    struct align_helper {
        char x;
        T t;
    };
    enum {value = sizeof(align_helper) - sizeof(T)};
};

// Metaprogram for determining if a type is the C++ "bool" or not
template<typename T> struct is_type_bool {enum {value = false};};
template<> struct is_type_bool<bool> {enum {value = true};};


} // namespace dynd

#endif // _DYND__TYPE_ID_HPP_
