//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <complex>

#include <dynd/config.hpp>

namespace dynd {
class string;

namespace ndt {
  class type;
}

enum type_kind_t {
  bool_kind,
  uint_kind,
  sint_kind,
  real_kind,
  complex_kind,
  void_kind,
  char_kind,

  // string_kind means subclass of base_string_type
  string_kind,
  bytes_kind,
  datetime_kind,

  // For type_type_id and other types that themselves represent types
  type_kind,

  // For any dimension types which have elements of all the same type
  dim_kind,

  // For structs
  struct_kind,
  // For tuples
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

  // For callables
  function_kind,

  // For symbolic types that represent a kind of type, like 'Any' or 'Fixed'
  kind_kind,
  // For symbolic types that represent a pattern, like 'T' or 'Dims... * R'
  pattern_kind,

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
  // A dynamic array type
  array_type_id,

  // blockref primitive types
  bytes_type_id,
  // A bytes buffer of a fixed size
  fixed_bytes_type_id,

  // A single string character
  char_type_id,

  // A variable-sized string type
  string_type_id,
  // A NULL-terminated string buffer of a fixed size
  fixed_string_type_id,

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

  // A fixed-sized strided array dimension type
  fixed_dim_type_id,
  // A variable-sized array dimension type
  var_dim_type_id,
  // A dimension made up of offsets
  //  offset_dim_type_id,

  // A struct type with variable layout
  struct_type_id,
  // A tuple type with variable layout
  tuple_type_id,
  option_type_id,

  // A type that enforces C contiguity
  c_contiguous_type_id,
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

  // Instances of this type are themselves types
  type_type_id,

  // Named symbolic types
  // Types matching a single type_kind_t, like Bool, UInt, SInt, Real, etc.
  kind_sym_type_id,
  // "Int", matching both UInt and SInt
  int_sym_type_id,
  // "Any", matching any type (dimensions and dtype)
  any_kind_type_id,
  // "Scalar" matchines any scalar type
  scalar_kind_type_id,

  // Symbolic types
  callable_type_id,
  typevar_type_id,
  typevar_dim_type_id,
  typevar_constructed_type_id,
  pow_dimsym_type_id,
  ellipsis_dim_type_id,
  // A special type which holds a fragment of canonical dimensions
  // for the purpose of broadcasting together named ellipsis type vars.
  dim_fragment_type_id,

  // The number of built-in, atomic types (including uninitialized and void)
  builtin_type_id_count = 19,

  // The number of types
  static_type_id_max = dim_fragment_type_id
};
#define DYND_BUILTIN_TYPE_ID_COUNT 19
#define DYND_TYPE_ID_MAX 100

template <type_id_t... I>
using type_id_sequence = integer_sequence<type_id_t, I...>;

typedef type_id_sequence<int8_type_id, int16_type_id, int32_type_id, int64_type_id, int128_type_id> int_type_ids;
typedef type_id_sequence<bool_type_id, uint8_type_id, uint16_type_id, uint32_type_id, uint64_type_id, uint128_type_id>
    uint_type_ids;
typedef type_id_sequence<float16_type_id, float32_type_id, float64_type_id, float128_type_id> float_type_ids;
typedef type_id_sequence<complex_float32_type_id, complex_float64_type_id> complex_type_ids;

typedef join<int_type_ids, uint_type_ids>::type integral_type_ids;
typedef join<integral_type_ids, join<float_type_ids, complex_type_ids>::type>::type arithmetic_type_ids;

typedef type_id_sequence<fixed_dim_type_id, var_dim_type_id> dim_type_ids;

inline void validate_type_id(type_id_t tp_id)
{
  if (tp_id > static_cast<type_id_t>(DYND_TYPE_ID_MAX)) {
    throw std::runtime_error("invalid type id");
  }
}

enum type_flags_t {
  // A symbolic name instead of just "0"
  type_flag_none = 0x00000000,
  // Memory of this type must be zero-initialized
  type_flag_zeroinit = 0x00000001,
  // Memory of this type must be constructed
  type_flag_construct = 0x00000002,
  // Instances of this type point into other memory
  // blocks, e.g. string_type, var_dim_type.
  type_flag_blockref = 0x00000004,
  // Memory of this type must be destroyed,
  // e.g. it might hold a reference count or similar state
  type_flag_destructor = 0x00000008,
  // Memory of this type is not readable directly from the host
  type_flag_not_host_readable = 0x00000010,
  // This type contains a symbolic construct like a type var
  type_flag_symbolic = 0x00000020,
  // This dimensions of this type are variadic (outermost dimensions, but not
  // dimensions within a struct, for example)
  type_flag_variadic = 0x00000040,
  // This type is indexable, either as a dimension or as a tuple / struct
  type_flag_indexable = 0x00000080,
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
  type_flags_operand_inherited = type_flag_zeroinit | type_flag_blockref | type_flag_construct | type_flag_destructor |
                                 type_flag_not_host_readable | type_flag_symbolic,
  // These are the flags expression types should inherit
  // from their value type
  type_flags_value_inherited = type_flag_symbolic | type_flag_variadic
};

DYND_API std::ostream &operator<<(std::ostream &o, type_kind_t kind);
DYND_API std::ostream &operator<<(std::ostream &o, type_id_t tid);

enum {
  /** A mask within which all the built-in type ids are guaranteed to fit */
  builtin_type_id_mask = 0x3f
};

// Forward declaration so we can make the is_builtin_type function here
namespace ndt {
  class base_type;
} // namespace dynd::nd

inline bool is_builtin_type(const ndt::base_type *dt)
{
  return reinterpret_cast<uintptr_t>(dt) < builtin_type_id_count;
}

namespace detail {
  // Simple metaprogram taking log base 2 of 1, 2, 4, and 8
  template <int I>
  struct log2_x;
  template <>
  struct log2_x<1> {
    enum { value = 0 };
  };
  template <>
  struct log2_x<2> {
    enum { value = 1 };
  };
  template <>
  struct log2_x<4> {
    enum { value = 2 };
  };
  template <>
  struct log2_x<8> {
    enum { value = 3 };
  };
}

// Type trait for the type id
template <typename T>
struct type_id_of;

template <typename T>
struct type_id_of<const T> {
  static const type_id_t value = type_id_of<T>::value;
};

// Can't use bool, because it doesn't have a guaranteed sizeof
template <>
struct type_id_of<bool1> {
  static const type_id_t value = bool_type_id;
};

template <>
struct type_id_of<char> {
  static const type_id_t value = ((char)-1) < 0 ? int8_type_id : uint8_type_id;
};

template <>
struct type_id_of<signed char> {
  static const type_id_t value = int8_type_id;
};

template <>
struct type_id_of<short> {
  static const type_id_t value = int16_type_id;
};

template <>
struct type_id_of<int> {
  static const type_id_t value = int32_type_id;
};

template <>
struct type_id_of<long> {
  static const type_id_t value = static_cast<type_id_t>(int8_type_id + detail::log2_x<sizeof(long)>::value);
};

template <>
struct type_id_of<long long> {
  static const type_id_t value = int64_type_id;
};

template <>
struct type_id_of<int128> {
  static const type_id_t value = int128_type_id;
};

template <>
struct type_id_of<uint8_t> {
  static const type_id_t value = uint8_type_id;
};

template <>
struct type_id_of<uint16_t> {
  static const type_id_t value = uint16_type_id;
};

template <>
struct type_id_of<unsigned int> {
  static const type_id_t value = uint32_type_id;
};

template <>
struct type_id_of<unsigned long> {
  static const type_id_t value = static_cast<type_id_t>(uint8_type_id + detail::log2_x<sizeof(unsigned long)>::value);
};

template <>
struct type_id_of<unsigned long long> {
  static const type_id_t value = uint64_type_id;
};

template <>
struct type_id_of<uint128> {
  static const type_id_t value = uint128_type_id;
};

template <>
struct type_id_of<float16> {
  static const type_id_t value = float16_type_id;
};

template <>
struct type_id_of<float32> {
  static const type_id_t value = float32_type_id;
};

template <>
struct type_id_of<float64> {
  static const type_id_t value = float64_type_id;
};

template <>
struct type_id_of<float128> {
  static const type_id_t value = float128_type_id;
};

template <>
struct type_id_of<complex64> {
  static const type_id_t value = complex_float32_type_id;
};

template <>
struct type_id_of<complex128> {
  static const type_id_t value = complex_float64_type_id;
};

template <>
struct type_id_of<void> {
  static const type_id_t value = void_type_id;
};

// Also allow type_id_of<std::complex<>> as synonyms for
// type_id_of<dynd_complex<>>
template <>
struct type_id_of<std::complex<float>> {
  static const type_id_t value = complex_float32_type_id;
};

template <>
struct type_id_of<std::complex<double>> {
  static const type_id_t value = complex_float64_type_id;
};

template <type_id_t TypeID>
struct type_of;

template <>
struct type_of<bool_type_id> {
  typedef bool1 type;
};
template <>
struct type_of<int8_type_id> {
  typedef int8 type;
};
template <>
struct type_of<int16_type_id> {
  typedef int16 type;
};
template <>
struct type_of<int32_type_id> {
  typedef int32 type;
};
template <>
struct type_of<int64_type_id> {
  typedef int64 type;
};
template <>
struct type_of<int128_type_id> {
  typedef int128 type;
};
template <>
struct type_of<uint8_type_id> {
  typedef uint8 type;
};
template <>
struct type_of<uint16_type_id> {
  typedef uint16 type;
};
template <>
struct type_of<uint32_type_id> {
  typedef uint32 type;
};
template <>
struct type_of<uint64_type_id> {
  typedef uint64 type;
};
template <>
struct type_of<uint128_type_id> {
  typedef uint128 type;
};
template <>
struct type_of<float16_type_id> {
  typedef float16 type;
};
template <>
struct type_of<float32_type_id> {
  typedef float32 type;
};
template <>
struct type_of<float64_type_id> {
  typedef float64 type;
};
template <>
struct type_of<float128_type_id> {
  typedef float128 type;
};
template <>
struct type_of<complex_float32_type_id> {
  typedef complex64 type;
};
template <>
struct type_of<complex_float64_type_id> {
  typedef complex128 type;
};

template <>
struct type_of<string_type_id> {
  typedef string type;
};

template <>
struct type_of<type_type_id> {
  typedef ndt::type type;
};

// Type trait for the kind
template <type_id_t TypeID>
struct type_kind_of;

template <>
struct type_kind_of<void_type_id> {
  static const type_kind_t value = void_kind;
};

template <>
struct type_kind_of<bool_type_id> {
  static const type_kind_t value = bool_kind;
};

template <>
struct type_kind_of<int8_type_id> {
  static const type_kind_t value = sint_kind;
};

template <>
struct type_kind_of<int16_type_id> {
  static const type_kind_t value = sint_kind;
};

template <>
struct type_kind_of<int32_type_id> {
  static const type_kind_t value = sint_kind;
};

template <>
struct type_kind_of<int64_type_id> {
  static const type_kind_t value = sint_kind;
};

template <>
struct type_kind_of<int128_type_id> {
  static const type_kind_t value = sint_kind;
};

template <>
struct type_kind_of<uint8_type_id> {
  static const type_kind_t value = uint_kind;
};

template <>
struct type_kind_of<uint16_type_id> {
  static const type_kind_t value = uint_kind;
};

template <>
struct type_kind_of<uint32_type_id> {
  static const type_kind_t value = uint_kind;
};

template <>
struct type_kind_of<uint64_type_id> {
  static const type_kind_t value = uint_kind;
};

template <>
struct type_kind_of<uint128_type_id> {
  static const type_kind_t value = uint_kind;
};

template <>
struct type_kind_of<float16_type_id> {
  static const type_kind_t value = real_kind;
};

template <>
struct type_kind_of<float32_type_id> {
  static const type_kind_t value = real_kind;
};

template <>
struct type_kind_of<float64_type_id> {
  static const type_kind_t value = real_kind;
};

template <>
struct type_kind_of<float128_type_id> {
  static const type_kind_t value = real_kind;
};

template <>
struct type_kind_of<complex_float32_type_id> {
  static const type_kind_t value = complex_kind;
};

template <>
struct type_kind_of<complex_float64_type_id> {
  static const type_kind_t value = complex_kind;
};

template <>
struct type_kind_of<fixed_dim_type_id> {
  static const type_kind_t value = dim_kind;
};

template <>
struct type_kind_of<var_dim_type_id> {
  static const type_kind_t value = dim_kind;
};

template <>
struct type_kind_of<pointer_type_id> {
  static const type_kind_t value = expr_kind;
};

template <>
struct type_kind_of<fixed_bytes_type_id> {
  static const type_kind_t value = bytes_kind;
};

template <>
struct type_kind_of<bytes_type_id> {
  static const type_kind_t value = bytes_kind;
};

template <>
struct type_kind_of<fixed_string_type_id> {
  static const type_kind_t value = string_kind;
};

template <>
struct type_kind_of<string_type_id> {
  static const type_kind_t value = string_kind;
};

template <>
struct type_kind_of<date_type_id> {
  static const type_kind_t value = datetime_kind;
};

template <>
struct type_kind_of<time_type_id> {
  static const type_kind_t value = datetime_kind;
};

template <>
struct type_kind_of<datetime_type_id> {
  static const type_kind_t value = datetime_kind;
};

template <>
struct type_kind_of<tuple_type_id> {
  static const type_kind_t value = tuple_kind;
};

template <>
struct type_kind_of<struct_type_id> {
  static const type_kind_t value = struct_kind;
};

template <>
struct type_kind_of<option_type_id> {
  static const type_kind_t value = option_kind;
};

namespace detail {

  template <type_id_t DstTypeID, type_kind_t DstTypeKind, type_id_t SrcTypeID, type_kind_t SrcTypeKind>
  struct is_lossless_assignable {
    static const bool value = false;
  };

  template <type_id_t DstTypeID, type_id_t SrcTypeID>
  struct is_lossless_assignable<DstTypeID, real_kind, SrcTypeID, sint_kind> {
    static const bool value = true;
  };

  template <type_id_t DstTypeID, type_id_t SrcTypeID>
  struct is_lossless_assignable<DstTypeID, real_kind, SrcTypeID, uint_kind> {
    static const bool value = true;
  };

  template <type_id_t DstTypeID, type_id_t SrcTypeID>
  struct is_lossless_assignable<DstTypeID, complex_kind, SrcTypeID, bool_kind> {
    static const bool value = true;
  };

  template <type_id_t DstTypeID, type_id_t SrcTypeID>
  struct is_lossless_assignable<DstTypeID, complex_kind, SrcTypeID, sint_kind> {
    static const bool value = false;
  };

  template <type_id_t DstTypeID, type_id_t SrcTypeID>
  struct is_lossless_assignable<DstTypeID, complex_kind, SrcTypeID, uint_kind> {
    static const bool value = false;
  };

  template <type_id_t DstTypeID, type_id_t SrcTypeID>
  struct is_lossless_assignable<DstTypeID, complex_kind, SrcTypeID, real_kind> {
    static const bool value = (sizeof(typename type_of<DstTypeID>::type) / 2) >
                              sizeof(typename type_of<SrcTypeID>::type);
  };

  template <type_id_t DstTypeID, type_id_t SrcTypeID, type_kind_t TypeKind>
  struct is_lossless_assignable<DstTypeID, TypeKind, SrcTypeID, TypeKind> {
    static const bool value = sizeof(typename type_of<DstTypeID>::type) > sizeof(typename type_of<SrcTypeID>::type);
  };

} // namespace dynd::detail

template <type_id_t DstTypeID, type_id_t Src0TypeID>
struct is_lossless_assignable : detail::is_lossless_assignable<DstTypeID, type_kind_of<DstTypeID>::value, Src0TypeID,
                                                               type_kind_of<Src0TypeID>::value> {
};

// Metaprogram for determining if a type is a valid C++ scalar
// of a particular type.
template <typename T>
struct is_dynd_scalar {
  enum { value = false };
};
template <>
struct is_dynd_scalar<bool1> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<char> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<signed char> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<short> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<int> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<long> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<long long> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<int128> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<unsigned char> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<unsigned short> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<unsigned int> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<unsigned long> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<unsigned long long> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<uint128> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<float16> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<float32> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<float64> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<float128> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<complex64> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<complex128> {
  enum { value = true };
};
// Allow std::complex as scalars equivalent to dynd_complex
template <>
struct is_dynd_scalar<std::complex<float>> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<std::complex<double>> {
  enum { value = true };
};

// Metaprogram for determining if a type is a valid C++ pointer to a scalar
// of a particular type.
template <typename T>
struct is_dynd_scalar_pointer {
  enum { value = false };
};
template <typename T>
struct is_dynd_scalar_pointer<T *> {
  enum { value = is_dynd_scalar<T>::value };
};

// Metaprogram for determining scalar alignment
template <typename T>
struct scalar_align_of {
  struct align_helper {
    char x;
    T t;
  };
  static const int value = sizeof(align_helper) - sizeof(T);
};

// Metaprogram for determining if a type is the C++ "bool" or not
template <typename T>
struct is_type_bool {
  enum { value = false };
};
template <>
struct is_type_bool<bool> {
  enum { value = true };
};

} // namespace dynd
