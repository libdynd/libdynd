#pragma once

#include <dynd/config.hpp>

namespace dynd {

namespace detail {

  // The check macros and the related templates are used to
  // define expression SFINAE based checks for whether or not
  // a given operator overload exists for specific input types.

  template <typename>
  struct sfinae_true : std::true_type {};

#define DYND_CHECK_UNARY_OP(OP, NAME)                                                                                  \
  template <typename T>                                                                                                \
  static auto NAME##_isdef_test(int DYND_UNUSED(a))->sfinae_true<decltype(OP std::declval<T>())>;                      \
                                                                                                                       \
  template <typename>                                                                                                  \
  static auto NAME##_isdef_test(long)->std::false_type;                                                                \
                                                                                                                       \
  template <typename Arg0Type>                                                                                         \
  struct isdef_##NAME : decltype(NAME##_isdef_test<Arg0Type>(0)) {};

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
  template <typename Arg0Type, typename Arg1Type>                                                                      \
  struct isdef_##NAME : decltype(NAME##_isdef_test<Arg0Type, Arg1Type>(0)) {};

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

  // logical_xor relies on the logical not operator.
  // Check if logical_xor is defined by checking if the logical not operator exists.

  // First define a template to check if a condition is true for every member of a
  // given parameter pack.

  template <template <typename> class Condition, typename... ArgTypes>
  struct all;

  template <template <typename> class Condition, typename Arg0Type>
  struct all<Condition, Arg0Type> {
    static constexpr bool value = Condition<Arg0Type>::value;
  };

  template <template <typename> class Condition, typename Arg0Type, typename... ArgTypes>
  struct all<Condition, Arg0Type, ArgTypes...> {
    static constexpr bool value = Condition<Arg0Type>::value && all<Condition, ArgTypes...>::value;
  };

  // Use a similar technique as before to check if a conversion to bool exists
  template <typename T>
  static auto bool_cast_isdef_test(int DYND_UNUSED(a)) -> sfinae_true<decltype(static_cast<bool>(std::declval<T>()))>;

  template <typename>
  static auto bool_cast_isdef_test(long) -> std::false_type;

  template <typename Arg0Type>
  struct isdef_bool_cast : decltype(bool_cast_isdef_test<Arg0Type>(0)) {};

  // logical_xor is defined if both types have a conversion to bool.
  template <typename Arg0Type, typename Arg1Type>
  using isdef_logical_xor = all<isdef_bool_cast, Arg0Type, Arg1Type>;

#define DYND_DEF_UNARY_OP_CALLABLE(OP, NAME)                                                                           \
  template <typename Arg0Type>                                                                                         \
  struct inline_##NAME {                                                                                               \
    static auto f(Arg0Type a) { return OP a; }                                                                         \
  };

  DYND_DEF_UNARY_OP_CALLABLE(+, plus)
  DYND_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_DEF_UNARY_OP_CALLABLE(-, minus)
  DYND_END_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_DEF_UNARY_OP_CALLABLE(!, logical_not)
  DYND_DEF_UNARY_OP_CALLABLE(~, bitwise_not)

#undef DYND_DEF_UNARY_OP_CALLABLE

#define DYND_DEF_BINARY_OP_CALLABLE(OP, NAME)                                                                          \
  template <typename Arg0Type, typename Arg1Type>                                                                      \
  struct inline_##NAME {                                                                                               \
    static auto f(Arg0Type a, Arg1Type b) { return a OP b; }                                                           \
  };

  DYND_DEF_BINARY_OP_CALLABLE(+, add)
  DYND_DEF_BINARY_OP_CALLABLE(-, subtract)
  DYND_DEF_BINARY_OP_CALLABLE(*, multiply)
  DYND_ALLOW_INT_BOOL_OPS
  DYND_DEF_BINARY_OP_CALLABLE(&, bitwise_and)
  DYND_END_ALLOW_INT_BOOL_OPS
  DYND_DEF_BINARY_OP_CALLABLE(&&, logical_and)
  DYND_ALLOW_INT_BOOL_OPS
  DYND_DEF_BINARY_OP_CALLABLE(|, bitwise_or)
  DYND_END_ALLOW_INT_BOOL_OPS
  DYND_DEF_BINARY_OP_CALLABLE(||, logical_or)
  DYND_ALLOW_INT_BOOL_OPS
  DYND_DEF_BINARY_OP_CALLABLE (^, bitwise_xor)
  DYND_DEF_BINARY_OP_CALLABLE(<<, left_shift)
  DYND_DEF_BINARY_OP_CALLABLE(>>, right_shift)
  DYND_END_ALLOW_INT_BOOL_OPS

#undef DYND_DEF_BINARY_OP_CALLABLE

  template <typename Arg0Type, typename Arg1Type>
  struct inline_logical_xor {
    static auto f(Arg0Type a, Arg1Type b) {
      DYND_ALLOW_INT_BOOL_CAST
      return static_cast<bool>(a) ^ static_cast<bool>(b);
      DYND_END_ALLOW_INT_BOOL_CAST
    }
  };

  template <typename Arg0Type, typename Arg1Type>
  struct inline_pow {
    DYND_ALLOW_INT_FLOAT_CAST
    static auto f(Arg0Type a, Arg1Type b) { return std::pow(static_cast<double>(a), static_cast<double>(b)); }
    DYND_END_ALLOW_INT_FLOAT_CAST
  };

  template <typename Arg0Type>
  struct inline_sqrt {
    static auto f(Arg0Type a) { return std::sqrt(a); }
  };

  template <typename Arg0Type>
  struct inline_cbrt {
    static auto f(Arg0Type a) { return std::cbrt(a); }
  };

  // Arithmetic operators that need zero checking.
  template <typename Arg0Type, typename Arg1Type>
  constexpr bool needs_zero_check() {
    return (std::is_same<Arg0Type, bool1>::value || is_signed_integral<Arg0Type>::value ||
            is_unsigned_integral<Arg0Type>::value) &&
           (std::is_same<Arg1Type, bool1>::value || is_signed_integral<Arg1Type>::value ||
            is_unsigned_integral<Arg0Type>::value);
  }

#define DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT(OP, NAME)                                                            \
  template <typename Arg0Type, typename Arg1Type, bool check>                                                          \
  struct inline_##NAME##_base;                                                                                         \
                                                                                                                       \
  template <typename Arg0Type, typename Arg1Type>                                                                      \
  struct inline_##NAME##_base<Arg0Type, Arg1Type, true> {                                                              \
    static auto f(Arg0Type a, Arg1Type b) {                                                                            \
      if (b == 0) {                                                                                                    \
        throw dynd::zero_division_error("Integer division or modulo by zero.");                                        \
      }                                                                                                                \
      return a OP b;                                                                                                   \
    }                                                                                                                  \
  };                                                                                                                   \
                                                                                                                       \
  template <typename Arg0Type, typename Arg1Type>                                                                      \
  struct inline_##NAME##_base<Arg0Type, Arg1Type, false> {                                                             \
    static auto f(Arg0Type a, Arg1Type b) { return a OP b; }                                                           \
  };                                                                                                                   \
                                                                                                                       \
  template <typename Arg0Type, typename Arg1Type>                                                                      \
  using inline_##NAME = inline_##NAME##_base<Arg0Type, Arg1Type, needs_zero_check<Arg0Type, Arg1Type>()>;

  DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT(/, divide)
  DYND_ALLOW_INT_BOOL_OPS
  DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT(%, mod)
  DYND_END_ALLOW_INT_BOOL_OPS

#undef DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT

} // namespace dynd::detail

namespace ndt {

  DYND_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_END_ALLOW_UNSIGNED_UNARY_MINUS

} // namespace dynd::ndt
} // namespace dynd
