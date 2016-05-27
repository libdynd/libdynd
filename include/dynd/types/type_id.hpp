//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <complex>
#include <iostream>

#include <dynd/config.hpp>

namespace dynd {
class bytes;
class string;

enum type_id_t {
  // The value zero is reserved for an uninitialized type.
  uninitialized_id,

  any_kind_id,    // "Any", matching any type (dimensions and dtype)
  scalar_kind_id, // "Scalar" matchines any scalar type

  bool_kind_id,
  // A 1-byte boolean type
  bool_id,

  // Signed integer types
  int_kind_id,
  int8_id,
  int16_id,
  int32_id,
  int64_id,
  int128_id,

  // Unsigned integer types
  uint_kind_id,
  uint8_id,
  uint16_id,
  uint32_id,
  uint64_id,
  uint128_id,

  // Floating point types
  float_kind_id,
  float16_id,
  float32_id,
  float64_id,
  float128_id,

  // Complex floating-point types
  complex_kind_id,
  complex_float32_id,
  complex_float64_id,

  // Means no type, just like in C. (Different from NumPy)
  void_id,

  dim_kind_id,

  bytes_kind_id,
  fixed_bytes_kind_id,
  fixed_bytes_id, // A bytes buffer of a fixed size
  bytes_id,       // blockref primitive types

  string_kind_id,
  fixed_string_kind_id,
  fixed_string_id, // A NULL-terminated string buffer of a fixed size
  char_id,         // A single string character
  string_id,       // A variable-sized string type

  // A tuple type with variable layout
  tuple_id,
  // A struct type with variable layout
  struct_id,

  fixed_dim_kind_id,
  fixed_dim_id, // A fixed-sized strided array dimension type
  var_dim_id,   // A variable-sized array dimension type
  // offset_dim_id, // A dimension made up of offsets

  categorical_id, // A categorical (enum-like) type
  option_id,
  pointer_id, // A pointer type
  memory_id,  // For types that specify a memory space

  type_id,  // Instances of this type are themselves types
  array_id, // A dynamic array type
  callable_id,

  expr_kind_id,
  adapt_id, // Adapter type
  expr_id,  // Advanced expression types

  // A CUDA host memory type
  cuda_host_id,
  // A CUDA device (global) memory type
  cuda_device_id,

  // A type for state in a functional
  state_id,

  // Symbolic types
  typevar_id,
  typevar_dim_id,
  typevar_constructed_id,
  pow_dimsym_id,
  ellipsis_dim_id,
  // A special type which holds a fragment of canonical dimensions
  // for the purpose of broadcasting together named ellipsis type vars.
  dim_fragment_id,
};

template <type_id_t Value>
using id_constant = std::integral_constant<type_id_t, Value>;

typedef type_sequence<int8_t, int16_t, int32_t, int64_t, int128> int_types;
typedef type_sequence<bool1, uint8_t, uint16_t, uint32_t, uint64_t, uint128> uint_types;
typedef type_sequence<float16, float, double, float128> float_types;
typedef type_sequence<complex<float>, complex<double>> complex_types;

typedef join<int_types, uint_types>::type integral_types;
typedef join<integral_types, join<float_types, complex_types>::type>::type arithmetic_types;

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

DYNDT_API std::ostream &operator<<(std::ostream &o, type_id_t tid);

// Forward declaration so we can make the is_builtin_type function here
namespace ndt {
  class base_type;
} // namespace dynd::nd

inline bool is_builtin_type(const ndt::base_type *dt) {
  switch (reinterpret_cast<uintptr_t>(dt)) {
  case uninitialized_id:
  case bool_id:
  case int8_id:
  case int16_id:
  case int32_id:
  case int64_id:
  case int128_id:
  case uint8_id:
  case uint16_id:
  case uint32_id:
  case uint64_id:
  case uint128_id:
  case float16_id:
  case float32_id:
  case float64_id:
  case float128_id:
  case complex_float32_id:
  case complex_float64_id:
  case void_id:
    return true;
  default:
    return false;
  }
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

namespace ndt {

  // Type trait for the type id
  template <typename T>
  struct id_of;

  // Can't use bool, because it doesn't have a guaranteed sizeof
  template <>
  struct id_of<bool1> {
    static const type_id_t value = bool_id;
  };

  template <>
  struct id_of<bool> : std::integral_constant<type_id_t, bool_id> {};

  template <>
  struct id_of<char> : std::integral_constant<type_id_t, ((char)-1) < 0 ? int8_id : uint8_id> {};

  template <>
  struct id_of<signed char> : std::integral_constant<type_id_t, int8_id> {};

  template <>
  struct id_of<short> : std::integral_constant<type_id_t, int16_id> {};

  template <>
  struct id_of<int> : std::integral_constant<type_id_t, int32_id> {};

  template <>
  struct id_of<long>
      : std::integral_constant<type_id_t, static_cast<type_id_t>(int8_id + detail::log2_x<sizeof(long)>::value)> {};

  template <>
  struct id_of<long long> : std::integral_constant<type_id_t, int64_id> {};

  template <>
  struct id_of<int128> : std::integral_constant<type_id_t, int128_id> {};

  template <>
  struct id_of<uint8_t> : std::integral_constant<type_id_t, uint8_id> {};

  template <>
  struct id_of<uint16_t> : std::integral_constant<type_id_t, uint16_id> {};

  template <>
  struct id_of<unsigned int> : std::integral_constant<type_id_t, uint32_id> {};

  template <>
  struct id_of<unsigned long>
      : std::integral_constant<type_id_t,
                               static_cast<type_id_t>(uint8_id + detail::log2_x<sizeof(unsigned long)>::value)> {};

  template <>
  struct id_of<unsigned long long> : std::integral_constant<type_id_t, uint64_id> {};

  template <>
  struct id_of<uint128> : std::integral_constant<type_id_t, uint128_id> {};

  template <>
  struct id_of<float16> : std::integral_constant<type_id_t, float16_id> {};

  template <>
  struct id_of<float32> : std::integral_constant<type_id_t, float32_id> {};

  template <>
  struct id_of<float64> : std::integral_constant<type_id_t, float64_id> {};

  template <>
  struct id_of<float128> : std::integral_constant<type_id_t, float128_id> {};

  template <>
  struct id_of<complex64> : std::integral_constant<type_id_t, complex_float32_id> {};

  template <>
  struct id_of<complex128> : std::integral_constant<type_id_t, complex_float64_id> {};

  template <>
  struct id_of<void> : std::integral_constant<type_id_t, void_id> {};

} // namespace dynd::ndt

template <type_id_t ID>
struct base_id_of;

template <>
struct base_id_of<scalar_kind_id> : id_constant<any_kind_id> {};

template <>
struct base_id_of<bool_kind_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<bool_id> : id_constant<bool_kind_id> {};

template <>
struct base_id_of<int_kind_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<int8_id> : id_constant<int_kind_id> {};

template <>
struct base_id_of<int16_id> : id_constant<int_kind_id> {};

template <>
struct base_id_of<int32_id> : id_constant<int_kind_id> {};

template <>
struct base_id_of<int64_id> : id_constant<int_kind_id> {};

template <>
struct base_id_of<int128_id> : id_constant<int_kind_id> {};

template <>
struct base_id_of<uint_kind_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<uint8_id> : id_constant<uint_kind_id> {};

template <>
struct base_id_of<uint16_id> : id_constant<uint_kind_id> {};

template <>
struct base_id_of<uint32_id> : id_constant<uint_kind_id> {};

template <>
struct base_id_of<uint64_id> : id_constant<uint_kind_id> {};

template <>
struct base_id_of<uint128_id> : id_constant<uint_kind_id> {};

template <>
struct base_id_of<float_kind_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<float16_id> : id_constant<float_kind_id> {};

template <>
struct base_id_of<float32_id> : id_constant<float_kind_id> {};

template <>
struct base_id_of<float64_id> : id_constant<float_kind_id> {};

template <>
struct base_id_of<float128_id> : id_constant<float_kind_id> {};

template <>
struct base_id_of<complex_kind_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<complex_float32_id> : id_constant<complex_kind_id> {};

template <>
struct base_id_of<complex_float64_id> : id_constant<complex_kind_id> {};

template <>
struct base_id_of<void_id> : id_constant<any_kind_id> {};

template <>
struct base_id_of<bytes_kind_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<fixed_bytes_kind_id> : id_constant<bytes_kind_id> {};

template <>
struct base_id_of<fixed_bytes_id> : id_constant<bytes_kind_id> {};

template <>
struct base_id_of<bytes_id> : id_constant<bytes_kind_id> {};

template <>
struct base_id_of<string_kind_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<fixed_string_kind_id> : id_constant<string_kind_id> {};

template <>
struct base_id_of<char_id> : id_constant<string_kind_id> {};

template <>
struct base_id_of<fixed_string_id> : id_constant<string_kind_id> {};

template <>
struct base_id_of<string_id> : id_constant<string_kind_id> {};

template <>
struct base_id_of<fixed_dim_kind_id> : id_constant<dim_kind_id> {};

template <>
struct base_id_of<fixed_dim_id> : id_constant<fixed_dim_kind_id> {};

template <>
struct base_id_of<var_dim_id> : id_constant<dim_kind_id> {};

template <>
struct base_id_of<pointer_id> : id_constant<any_kind_id> {};

template <>
struct base_id_of<memory_id> : id_constant<any_kind_id> {};

template <>
struct base_id_of<expr_kind_id> : id_constant<any_kind_id> {};

template <>
struct base_id_of<adapt_id> : id_constant<expr_kind_id> {};

template <>
struct base_id_of<tuple_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<struct_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<option_id> : id_constant<any_kind_id> {};

template <>
struct base_id_of<categorical_id> : id_constant<any_kind_id> {};

template <>
struct base_id_of<expr_id> : id_constant<any_kind_id> {};

template <>
struct base_id_of<type_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<callable_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<array_id> : id_constant<scalar_kind_id> {};

template <>
struct base_id_of<dim_kind_id> : id_constant<any_kind_id> {};

template <>
struct base_id_of<typevar_id> : id_constant<scalar_kind_id> {};

template <typename ReturnType, typename Arg0Type, typename Enable = void>
struct is_lossless_assignable {
  static const bool value = false;
};

template <typename ReturnType, typename Arg0Type>
struct is_lossless_assignable<ReturnType, Arg0Type, std::enable_if_t<is_floating_point<ReturnType>::value &&
                                                                     is_signed_integral<Arg0Type>::value>> {
  static const bool value = true;
};

template <typename ReturnType, typename Arg0Type>
struct is_lossless_assignable<ReturnType, Arg0Type, std::enable_if_t<is_floating_point<ReturnType>::value &&
                                                                     is_unsigned_integral<Arg0Type>::value>> {
  static const bool value = true;
};

template <typename ReturnType>
struct is_lossless_assignable<ReturnType, bool1, std::enable_if_t<is_complex<ReturnType>::value>> {
  static const bool value = true;
};

template <typename ReturnType, typename Arg0Type>
struct is_lossless_assignable<ReturnType, Arg0Type,
                              std::enable_if_t<is_complex<ReturnType>::value && is_signed_integral<Arg0Type>::value>> {
  static const bool value = false;
};

template <typename ReturnType, typename Arg0Type>
struct is_lossless_assignable<
    ReturnType, Arg0Type, std::enable_if_t<is_complex<ReturnType>::value && is_unsigned_integral<Arg0Type>::value>> {
  static const bool value = false;
};

template <typename ReturnType, typename Arg0Type>
struct is_lossless_assignable<ReturnType, Arg0Type,
                              std::enable_if_t<is_complex<ReturnType>::value && is_floating_point<Arg0Type>::value>> {
  static const bool value = (sizeof(ReturnType) / 2) > sizeof(Arg0Type);
};

template <typename ReturnType, typename Arg0Type>
struct is_lossless_assignable<ReturnType, Arg0Type, std::enable_if_t<is_signed_integral<ReturnType>::value &&
                                                                     is_signed_integral<Arg0Type>::value>> {
  static const bool value = sizeof(ReturnType) > sizeof(Arg0Type);
};

template <typename ReturnType, typename Arg0Type>
struct is_lossless_assignable<ReturnType, Arg0Type, std::enable_if_t<is_unsigned_integral<ReturnType>::value &&
                                                                     is_unsigned_integral<Arg0Type>::value>> {
  static const bool value = sizeof(ReturnType) > sizeof(Arg0Type);
};

template <typename ReturnType, typename Arg0Type>
struct is_lossless_assignable<ReturnType, Arg0Type, std::enable_if_t<is_floating_point<ReturnType>::value &&
                                                                     is_floating_point<Arg0Type>::value>> {
  static const bool value = sizeof(ReturnType) > sizeof(Arg0Type);
};

template <typename ReturnType, typename Arg0Type>
struct is_lossless_assignable<ReturnType, Arg0Type,
                              std::enable_if_t<is_complex<ReturnType>::value && is_complex<Arg0Type>::value>> {
  static const bool value = sizeof(ReturnType) > sizeof(Arg0Type);
};

// Metaprogram for determining if a type is a valid C++ scalar
// of a particular type.
template <typename T>
struct is_dynd_scalar {
  enum { value = false };
};
template <>
struct is_dynd_scalar<bool> {
  enum { value = true };
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

// Get the type id for types that are valid for describing the return value of
// a type property.
template <typename T>
struct property_type_id_of {
  static const type_id_t value = ndt::id_of<T>::value;
};

template <>
struct property_type_id_of<std::string> {
  static const type_id_t value = string_id;
};

} // namespace dynd
