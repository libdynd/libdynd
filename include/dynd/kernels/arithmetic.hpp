#pragma once

#include <dynd/config.hpp>

namespace dynd {

namespace detail {
  template <typename>
  struct sfinae_true : std::true_type {};

#define DYND_CHECK_UNARY_OP(OP, NAME)                                                                                  \
  template <typename T>                                                                                                \
  static auto NAME##_isdef_test(int DYND_UNUSED(a))->sfinae_true<decltype(OP std::declval<T>())>;                      \
                                                                                                                       \
  template <typename>                                                                                                  \
  static auto NAME##_isdef_test(long)->std::false_type;                                                                \
                                                                                                                       \
  template <type_id_t Src0TypeID>                                                                                      \
  struct isdef_##NAME : decltype(NAME##_isdef_test<typename type_of<Src0TypeID>::type>(0)) {};

  DYND_CHECK_UNARY_OP(+, plus)
  DYND_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_CHECK_UNARY_OP(-, minus)
  DYND_END_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_CHECK_UNARY_OP(!, logical_not)
  DYND_CHECK_UNARY_OP(~, bitwise_not)

#undef DYND_CHECK_UNARY_OP

#define DYND_CHECK_BINARY_OP(OP, NAME)                                                                                 \
                                                                                                                       \
  template <typename T, typename U>                                                                                    \
  static auto NAME##_isdef_test(int DYND_UNUSED(a))->sfinae_true<decltype(std::declval<T>() OP std::declval<U>())>;    \
                                                                                                                       \
  template <typename, typename>                                                                                        \
  static auto NAME##_isdef_test(long)->std::false_type;                                                                \
                                                                                                                       \
  template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                                \
  struct isdef_##NAME                                                                                                  \
      : decltype(NAME##_isdef_test<typename type_of<Src0TypeID>::type, typename type_of<Src1TypeID>::type>(0)) {};

  DYND_CHECK_BINARY_OP(+, add)
  DYND_CHECK_BINARY_OP(-, subtract)
  DYND_CHECK_BINARY_OP(*, multiply)
  DYND_CHECK_BINARY_OP(/, divide)
  DYND_CHECK_BINARY_OP(%, mod)
  DYND_CHECK_BINARY_OP(&, bitwise_and)
  DYND_CHECK_BINARY_OP(&&, logical_and)
  DYND_CHECK_BINARY_OP(|, bitwise_or)
  DYND_CHECK_BINARY_OP(||, logical_or)
  DYND_CHECK_BINARY_OP (^, bitwise_xor)
  DYND_CHECK_BINARY_OP(<<, left_shift)
  DYND_CHECK_BINARY_OP(>>, right_shift)

#undef DYND_CHECK_BINARY_OP

#define DYND_DEF_UNARY_OP_CALLABLE(OP, NAME)                                                                           \
  template <type_id_t Src0TypeID>                                                                                      \
  struct inline_##NAME {                                                                                               \
    static auto f(typename type_of<Src0TypeID>::type a) { return OP a; }                                               \
  };

  DYND_DEF_UNARY_OP_CALLABLE(+, plus)
  DYND_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_DEF_UNARY_OP_CALLABLE(-, minus)
  DYND_END_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_DEF_UNARY_OP_CALLABLE(!, logical_not)
  DYND_DEF_UNARY_OP_CALLABLE(~, bitwise_not)

#undef DYND_DEF_UNARY_OP_CALLABLE

#define DYND_DEF_BINARY_OP_CALLABLE(OP, NAME)                                                                          \
  template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                                \
  struct inline_##NAME {                                                                                               \
    static auto f(typename type_of<Src0TypeID>::type a, typename type_of<Src1TypeID>::type b) { return a OP b; }       \
  };

  DYND_DEF_BINARY_OP_CALLABLE(+, add)
  DYND_DEF_BINARY_OP_CALLABLE(-, subtract)
  DYND_DEF_BINARY_OP_CALLABLE(*, multiply)
  DYND_ALLOW_INT_BOOL_OPS
  DYND_DEF_BINARY_OP_CALLABLE(&, bitwise_and)
  DYND_END_ALLOW_INT_BOOL_OPS
  DYND_DEF_BINARY_OP_CALLABLE(&&, logical_and)
  DYND_DEF_BINARY_OP_CALLABLE(|, bitwise_or)
  DYND_DEF_BINARY_OP_CALLABLE(||, logical_or)
  DYND_DEF_BINARY_OP_CALLABLE (^, bitwise_xor)
  DYND_DEF_BINARY_OP_CALLABLE(<<, left_shift)
  DYND_DEF_BINARY_OP_CALLABLE(>>, right_shift)

#undef DYND_DEF_BINARY_OP_CALLABLE

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct inline_logical_xor {
    static auto f(typename type_of<Src0TypeID>::type a, typename type_of<Src1TypeID>::type b) { return (!a) ^ (!b); }
  };

  // For the time being, with all internal types, the result of the logical not operator should always be
  // a boolean value, so we can just check if the logical not operator is defined.
  // If that is no longer true at some point, we can give logical_xor its own
  // expression SFINAE based test for existence.

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  constexpr bool needs_zero_check() {
    using Base0 = base_id_of<Src0TypeID>;
    using Base1 = base_id_of<Src1TypeID>;
    return ((Base0::value == bool_kind_id) || (Base0::value == int_kind_id) || (Base0::value == uint_kind_id)) &&
           ((Base1::value == bool_kind_id) || (Base1::value == int_kind_id) || (Base1::value == uint_kind_id));
  }

#define DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT(OP, NAME)                                                            \
  template <type_id_t Src0TypeID, type_id_t Src1TypeID, bool check>                                                    \
  struct inline_##NAME##_base;                                                                                         \
                                                                                                                       \
  template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                                \
  struct inline_##NAME##_base<Src0TypeID, Src1TypeID, true> {                                                          \
    static auto f(typename type_of<Src0TypeID>::type a, typename type_of<Src1TypeID>::type b) {                        \
      if (b == 0) {                                                                                                    \
        throw dynd::zero_division_error("Integer division or modulo by zero.");                                        \
      }                                                                                                                \
      return a OP b;                                                                                                   \
    }                                                                                                                  \
  };                                                                                                                   \
                                                                                                                       \
  template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                                \
  struct inline_##NAME##_base<Src0TypeID, Src1TypeID, false> {                                                         \
    static auto f(typename type_of<Src0TypeID>::type a, typename type_of<Src1TypeID>::type b) { return a OP b; }       \
  };                                                                                                                   \
                                                                                                                       \
  template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                                \
  using inline_##NAME = inline_##NAME##_base<Src0TypeID, Src1TypeID, needs_zero_check<Src0TypeID, Src1TypeID>()>;

  DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT(/, divide)
  DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT(%, mod)

#undef DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT

} // namespace dynd::detail

namespace ndt {

  DYND_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_END_ALLOW_UNSIGNED_UNARY_MINUS

} // namespace dynd::ndt
} // namespace dynd
