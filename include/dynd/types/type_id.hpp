//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <complex>

#include <dynd/config.hpp>
#include <dynd/types/dynd_int128.hpp>
#include <dynd/types/dynd_uint128.hpp>
#include <dynd/types/dynd_float16.hpp>
#include <dynd/types/dynd_float128.hpp>
#include <dynd/types/complex.hpp>

namespace dynd {

enum type_kind_t {
  bool_kind,
  uint_kind,
  int_kind,
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

  // For arrfuncs
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

  // A fixed-sized strided array dimension type
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

  // Named symbolic types
  // "Fixed", an symbolic fixed array dimension type
  fixed_dimsym_type_id,
  // "Any", matching any type (dimensions and dtype)
  any_sym_type_id,

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

  // Symbolic types
  arrfunc_type_id,
  typevar_type_id,
  typevar_dim_type_id,
  typevar_constructed_type_id,
  pow_dimsym_type_id,
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
  // The (outermost) type should be considered as a dimension
  type_flag_dim = 0x00000002,
  // Memory of this type must be zero-initialized
  type_flag_zeroinit = 0x00000004,
  // Memory of this type must be constructed
  // type_flag_constructor = 0x00000008,
  // Instances of this type point into other memory
  // blocks, e.g. string_type, var_dim_type.
  type_flag_blockref = 0x00000010,
  // Memory of this type must be destroyed,
  // e.g. it might hold a reference count or similar state
  type_flag_destructor = 0x00000020,
  // Memory of this type is not readable directly from the host
  type_flag_not_host_readable = 0x00000040,
  // This type contains a symbolic construct like a type var
  type_flag_symbolic = 0x00000080,
  // This dimensions of this type are variadic (outermost dimensions, but not
  // dimensions within a struct, for example)
  type_flag_variadic = 0x00000100,
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
      type_flag_zeroinit | type_flag_blockref | type_flag_destructor |
      type_flag_not_host_readable | type_flag_symbolic,
  // These are the flags expression types should inherit
  // from their value type
  type_flags_value_inherited =
      type_flag_scalar | type_flag_symbolic | type_flag_variadic
};

std::ostream &operator<<(std::ostream &o, type_kind_t kind);
std::ostream &operator<<(std::ostream &o, type_id_t tid);

enum {
  /** A mask within which all the built-in type ids are guaranteed to fit */
  builtin_type_id_mask = 0x3f
};

// Forward declaration so we can make the is_builtin_type function here
class base_type;

inline bool is_builtin_type(const base_type *dt)
{
  return reinterpret_cast<uintptr_t>(dt) < builtin_type_id_count;
}

// A boolean class for dynamicndarray which is one-byte big
class dynd_bool {
  char m_value;

public:
  DYND_CUDA_HOST_DEVICE dynd_bool() : m_value(0) {}

  DYND_CUDA_HOST_DEVICE dynd_bool(bool value) : m_value(value) {}

  // Special case complex conversion to avoid ambiguous overload
  template <class T>
  DYND_CUDA_HOST_DEVICE dynd_bool(complex<T> value)
      : m_value(value != complex<T>(0))
  {
  }

  DYND_CUDA_HOST_DEVICE operator bool() const { return m_value != 0; }
};

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
  enum { value = type_id_of<T>::value };
};

// Can't use bool, because it doesn't have a guaranteed sizeof
template <>
struct type_id_of<dynd_bool> {
  enum { value = bool_type_id };
};
template <>
struct type_id_of<char> {
  enum { value = ((char)-1) < 0 ? int8_type_id : uint8_type_id };
};
template <>
struct type_id_of<signed char> {
  enum { value = int8_type_id };
};
template <>
struct type_id_of<short> {
  enum { value = int16_type_id };
};
template <>
struct type_id_of<int> {
  enum { value = int32_type_id };
};
template <>
struct type_id_of<long> {
  enum { value = int8_type_id + detail::log2_x<sizeof(long)>::value };
};
template <>
struct type_id_of<long long> {
  enum { value = int64_type_id };
};
template <>
struct type_id_of<dynd_int128> {
  enum { value = int128_type_id };
};
template <>
struct type_id_of<uint8_t> {
  enum { value = uint8_type_id };
};
template <>
struct type_id_of<uint16_t> {
  enum { value = uint16_type_id };
};
template <>
struct type_id_of<unsigned int> {
  enum { value = uint32_type_id };
};
template <>
struct type_id_of<unsigned long> {
  enum { value = uint8_type_id + detail::log2_x<sizeof(unsigned long)>::value };
};
template <>
struct type_id_of<unsigned long long> {
  enum { value = uint64_type_id };
};
template <>
struct type_id_of<dynd_uint128> {
  enum { value = uint128_type_id };
};
template <>
struct type_id_of<dynd_float16> {
  enum { value = float16_type_id };
};
template <>
struct type_id_of<float> {
  enum { value = float32_type_id };
};
template <>
struct type_id_of<double> {
  enum { value = float64_type_id };
};
template <>
struct type_id_of<dynd_float128> {
  enum { value = float128_type_id };
};
template <>
struct type_id_of<complex<float>> {
  enum { value = complex_float32_type_id };
};
template <>
struct type_id_of<complex<double>> {
  enum { value = complex_float64_type_id };
};
template <>
struct type_id_of<void> {
  enum { value = void_type_id };
};
// Also allow type_id_of<std::complex<>> as synonyms for
// type_id_of<dynd_complex<>>
template <>
struct type_id_of<std::complex<float>> {
  enum { value = complex_float32_type_id };
};
template <>
struct type_id_of<std::complex<double>> {
  enum { value = complex_float64_type_id };
};

template <type_id_t type_id>
struct type_of;

template <>
struct type_of<int8_type_id> {
  typedef int8_t type;
};
template <>
struct type_of<int16_type_id> {
  typedef int16_t type;
};
template <>
struct type_of<int32_type_id> {
  typedef int type;
};
template <>
struct type_of<int64_type_id> {
  typedef long long type;
};
template <>
struct type_of<int128_type_id> {
  typedef dynd_int128 type;
};
template <>
struct type_of<uint8_type_id> {
  typedef uint8_t type;
};
template <>
struct type_of<uint16_type_id> {
  typedef uint16_t type;
};
template <>
struct type_of<uint32_type_id> {
  typedef unsigned type;
};
template <>
struct type_of<uint64_type_id> {
  typedef unsigned long long type;
};
template <>
struct type_of<uint128_type_id> {
  typedef dynd_uint128 type;
};
template <>
struct type_of<float16_type_id> {
  typedef dynd_float16 type;
};
template <>
struct type_of<float32_type_id> {
  typedef float type;
};
template <>
struct type_of<float64_type_id> {
  typedef double type;
};
template <>
struct type_of<float128_type_id> {
  typedef dynd_float128 type;
};
template <>
struct type_of<complex_float32_type_id> {
  typedef complex<float> type;
};
template <>
struct type_of<complex_float64_type_id> {
  typedef complex<double> type;
};

// Type trait for the kind
template <typename T>
struct dynd_kind_of;

template <>
struct dynd_kind_of<void> {
  static const type_kind_t value = void_kind;
};
// Can't use bool, because it doesn't have a guaranteed sizeof
template <>
struct dynd_kind_of<dynd_bool> {
  static const type_kind_t value = bool_kind;
};
template <>
struct dynd_kind_of<char> {
  static const type_kind_t value = ((char)-1) < 0 ? int_kind : uint_kind;
};
template <>
struct dynd_kind_of<signed char> {
  static const type_kind_t value = int_kind;
};
template <>
struct dynd_kind_of<short> {
  static const type_kind_t value = int_kind;
};
template <>
struct dynd_kind_of<int> {
  static const type_kind_t value = int_kind;
};
template <>
struct dynd_kind_of<long> {
  static const type_kind_t value = int_kind;
};
template <>
struct dynd_kind_of<long long> {
  static const type_kind_t value = int_kind;
};
template <>
struct dynd_kind_of<dynd_int128> {
  static const type_kind_t value = int_kind;
};
template <>
struct dynd_kind_of<uint8_t> {
  static const type_kind_t value = uint_kind;
};
template <>
struct dynd_kind_of<uint16_t> {
  static const type_kind_t value = uint_kind;
};
template <>
struct dynd_kind_of<unsigned int> {
  static const type_kind_t value = uint_kind;
};
template <>
struct dynd_kind_of<unsigned long> {
  static const type_kind_t value = uint_kind;
};
template <>
struct dynd_kind_of<unsigned long long> {
  static const type_kind_t value = uint_kind;
};
template <>
struct dynd_kind_of<dynd_uint128> {
  static const type_kind_t value = uint_kind;
};
template <>
struct dynd_kind_of<dynd_float16> {
  static const type_kind_t value = real_kind;
};
template <>
struct dynd_kind_of<float> {
  static const type_kind_t value = real_kind;
};
template <>
struct dynd_kind_of<double> {
  static const type_kind_t value = real_kind;
};
template <>
struct dynd_kind_of<dynd_float128> {
  static const type_kind_t value = real_kind;
};
template <typename T>
struct dynd_kind_of<complex<T>> {
  static const type_kind_t value = complex_kind;
};

// Metaprogram for determining if a type is a valid C++ scalar
// of a particular type.
template <typename T>
struct is_dynd_scalar {
  enum { value = false };
};
template <>
struct is_dynd_scalar<dynd_bool> {
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
struct is_dynd_scalar<dynd_int128> {
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
struct is_dynd_scalar<dynd_uint128> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<dynd_float16> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<float> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<double> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<dynd_float128> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<complex<float>> {
  enum { value = true };
};
template <>
struct is_dynd_scalar<complex<double>> {
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
  enum { value = sizeof(align_helper) - sizeof(T) };
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

namespace detail {

  template <typename T, int N>
  class array_by_type_id;

  template <typename T>
  class array_by_type_id<T, 1> {
    T m_data[builtin_type_id_count];

  public:
    array_by_type_id(const std::initializer_list<std::pair<type_id_t, T>> &data)
    {
      for (const std::pair<type_id_t, T> &pair : data) {
        m_data[pair.first] = pair.second;
      }
    }

    T *data() { return m_data; }
    const T *data() const { return m_data; }

    T &at(type_id_t i) { return m_data[i]; }
    const T &at(type_id_t i) const { return m_data[i]; }

    T &operator()(type_id_t i) { return at(i); }
    const T &operator()(type_id_t i) const { return at(i); }
  };

  template <typename T>
  class array_by_type_id<T, 2> {
    T m_data[builtin_type_id_count][builtin_type_id_count];

  public:
    array_by_type_id() {}

    array_by_type_id(const std::initializer_list<
        std::pair<std::pair<type_id_t, type_id_t>, T>> &data)
    {
      for (const std::pair<std::pair<type_id_t, type_id_t>, T> &pair : data) {
        m_data[pair.first.first][pair.first.second] = pair.second;
      }
    }

    T *data() { return m_data; }
    const T *data() const { return m_data; }

    T &at(type_id_t i, type_id_t j) { return m_data[i][j]; }
    const T &at(type_id_t i, type_id_t j) const { return m_data[i][j]; }

    T &operator()(type_id_t i, type_id_t j) { return at(i, j); }
    const T &operator()(type_id_t i, type_id_t j) const { return at(i, j); }
  };

} // namespace dynd::detail
} // namespace dynd