#pragma once

#include <dynd/kernels/apply.hpp>
#include <dynd/option.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {

namespace detail {
  template <typename>
  struct sfinae_true : std::true_type {
  };

#define DYND_CHECK_UNARY_OP(OP, NAME)                                                                                  \
  template <typename T>                                                                                                \
  static auto NAME##_isdef_test(int DYND_UNUSED(a))->sfinae_true<decltype(OP std::declval<T>())>;                      \
                                                                                                                       \
  template <typename>                                                                                                  \
  static auto NAME##_isdef_test(long)->std::false_type;                                                                \
                                                                                                                       \
  template <type_id_t Src0TypeID>                                                                                      \
  struct isdef_##NAME : decltype(NAME##_isdef_test<typename type_of<Src0TypeID>::type>(0)) {                           \
  };

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
      : decltype(NAME##_isdef_test<typename type_of<Src0TypeID>::type, typename type_of<Src1TypeID>::type>(0)) {       \
  };

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

} // namespace dynd::detail

namespace nd {

#define DYND_DEF_UNARY_OP_CALLABLE(OP, NAME)                                                                           \
  namespace detail {                                                                                                   \
    template <type_id_t Src0TypeID>                                                                                    \
    struct inline_##NAME {                                                                                             \
      static auto f(typename type_of<Src0TypeID>::type a) { return OP a; }                                             \
    };                                                                                                                 \
                                                                                                                       \
    template <type_id_t Src0TypeID, bool Defined = dynd::detail::isdef_##NAME<Src0TypeID>::value>                      \
    class NAME##_callable                                                                                              \
        : public functional::apply_function_callable<decltype(&detail::inline_##NAME<Src0TypeID>::f),                  \
                                                     &detail::inline_##NAME<Src0TypeID>::f> {                          \
    public:                                                                                                            \
      NAME##_callable()                                                                                                \
          : functional::apply_function_callable<decltype(&detail::inline_##NAME<Src0TypeID>::f),                       \
                                                &detail::inline_##NAME<Src0TypeID>::f>(                                \
                ndt::make_type<decltype(dynd::nd::detail::inline_##NAME<Src0TypeID>::f)>())                            \
      {                                                                                                                \
      }                                                                                                                \
    };                                                                                                                 \
  } /* namespace detail */                                                                                             \
  template <type_id_t Src0TypeID>                                                                                      \
  using NAME##_callable = detail::NAME##_callable<Src0TypeID>;

  DYND_DEF_UNARY_OP_CALLABLE(+, plus)
  DYND_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_DEF_UNARY_OP_CALLABLE(-, minus)
  DYND_END_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_DEF_UNARY_OP_CALLABLE(!, logical_not)
  DYND_DEF_UNARY_OP_CALLABLE(~, bitwise_not)

#undef DYND_DEF_UNARY_OP_CALLABLE

#define DYND_DEF_BINARY_OP_CALLABLE(OP, NAME)                                                                          \
  namespace detail {                                                                                                   \
    template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                              \
    struct inline_##NAME {                                                                                             \
      static auto f(typename type_of<Src0TypeID>::type a, typename type_of<Src1TypeID>::type b) { return a OP b; }     \
    };                                                                                                                 \
    template <type_id_t Src0TypeID, type_id_t Src1TypeID,                                                              \
              bool Defined = dynd::detail::isdef_##NAME<Src0TypeID, Src1TypeID>::value>                                \
    class NAME##_callable                                                                                              \
        : public functional::apply_function_callable<decltype(&detail::inline_##NAME<Src0TypeID, Src1TypeID>::f),      \
                                                     &detail::inline_##NAME<Src0TypeID, Src1TypeID>::f> {              \
    public:                                                                                                            \
      NAME##_callable()                                                                                                \
          : functional::apply_function_callable<decltype(&detail::inline_##NAME<Src0TypeID, Src1TypeID>::f),           \
                                                &detail::inline_##NAME<Src0TypeID, Src1TypeID>::f>(                    \
                ndt::make_type<decltype(dynd::nd::detail::inline_##NAME<Src0TypeID, Src1TypeID>::f)>())                \
      {                                                                                                                \
      }                                                                                                                \
    };                                                                                                                 \
  } /* namespace detail */                                                                                             \
                                                                                                                       \
  template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                                \
  using NAME##_callable = detail::NAME##_callable<Src0TypeID, Src1TypeID>;

  DYND_DEF_BINARY_OP_CALLABLE(+, add)
  DYND_DEF_BINARY_OP_CALLABLE(-, subtract)
  DYND_DEF_BINARY_OP_CALLABLE(*, multiply)
  DYND_DEF_BINARY_OP_CALLABLE(&, bitwise_and)
  DYND_DEF_BINARY_OP_CALLABLE(&&, logical_and)
  DYND_DEF_BINARY_OP_CALLABLE(|, bitwise_or)
  DYND_DEF_BINARY_OP_CALLABLE(||, logical_or)
  DYND_DEF_BINARY_OP_CALLABLE (^, bitwise_xor)
  DYND_DEF_BINARY_OP_CALLABLE(<<, left_shift)
  DYND_DEF_BINARY_OP_CALLABLE(>>, right_shift)

#undef DYND_DEF_BINARY_OP_CALLABLE

  namespace detail {
    template <type_id_t Src0TypeID, type_id_t Src1TypeID>
    struct inline_logical_xor {
      static auto f(typename type_of<Src0TypeID>::type a, typename type_of<Src1TypeID>::type b) { return (!a) ^ (!b); }
    };

    // For the time being, with all internal types, the result of the logical not operator should always be
    // a boolean value, so we can just check if the logical not operator is defined.
    // If that is no longer true at some point, we can give logical_xor its own
    // expression SFINAE based test for existence.
    template <type_id_t Src0TypeID, type_id_t Src1TypeID, bool Defined =
                                                              dynd::detail::isdef_logical_not<Src0TypeID>::value
                                                                  &&dynd::detail::isdef_logical_not<Src1TypeID>::value>
    class logical_xor_callable
        : public functional::apply_function_callable<decltype(&detail::inline_logical_xor<Src0TypeID, Src1TypeID>::f),
                                                     &detail::inline_logical_xor<Src0TypeID, Src1TypeID>::f> {
    public:
      logical_xor_callable()
          : functional::apply_function_callable<decltype(&detail::inline_logical_xor<Src0TypeID, Src1TypeID>::f),
                                                &detail::inline_logical_xor<Src0TypeID, Src1TypeID>::f>(
                ndt::make_type<decltype(&dynd::nd::detail::inline_logical_xor<Src0TypeID, Src1TypeID>::f)>())
      {
      }
    };
  } /* namespace detail */
  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  using logical_xor_callable = detail::logical_xor_callable<Src0TypeID, Src1TypeID>;

  namespace detail {
    template <type_id_t Src0TypeID, type_id_t Src1TypeID>
    constexpr bool needs_zero_check()
    {
      using Base0 = base_id_of<Src0TypeID>;
      using Base1 = base_id_of<Src1TypeID>;
      return ((Base0::value == bool_kind_id) || (Base0::value == int_kind_id) || (Base0::value == uint_kind_id)) &&
             ((Base1::value == bool_kind_id) || (Base1::value == int_kind_id) || (Base1::value == uint_kind_id));
    }
  }

#define DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT(OP, NAME)                                                            \
  namespace detail {                                                                                                   \
    template <type_id_t Src0TypeID, type_id_t Src1TypeID, bool check>                                                  \
    struct inline_##NAME##_base;                                                                                       \
                                                                                                                       \
    template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                              \
    struct inline_##NAME##_base<Src0TypeID, Src1TypeID, true> {                                                        \
      static auto f(typename type_of<Src0TypeID>::type a, typename type_of<Src1TypeID>::type b)                        \
      {                                                                                                                \
        if (b == 0) {                                                                                                  \
          throw dynd::zero_division_error("Integer division or modulo by zero.");                                      \
        }                                                                                                              \
        return a OP b;                                                                                                 \
      }                                                                                                                \
    };                                                                                                                 \
                                                                                                                       \
    template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                              \
    struct inline_##NAME##_base<Src0TypeID, Src1TypeID, false> {                                                       \
      static auto f(typename type_of<Src0TypeID>::type a, typename type_of<Src1TypeID>::type b) { return a OP b; }     \
    };                                                                                                                 \
                                                                                                                       \
    template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                              \
    using inline_##NAME = inline_##NAME##_base<Src0TypeID, Src1TypeID, needs_zero_check<Src0TypeID, Src1TypeID>()>;    \
                                                                                                                       \
    template <type_id_t Src0TypeID, type_id_t Src1TypeID,                                                              \
              bool Defined = dynd::detail::isdef_##NAME<Src0TypeID, Src1TypeID>::value>                                \
    class NAME##_callable                                                                                              \
        : public functional::apply_function_callable<decltype(&detail::inline_##NAME<Src0TypeID, Src1TypeID>::f),      \
                                                     &detail::inline_##NAME<Src0TypeID, Src1TypeID>::f> {              \
    public:                                                                                                            \
      NAME##_callable()                                                                                                \
          : functional::apply_function_callable<decltype(&detail::inline_##NAME<Src0TypeID, Src1TypeID>::f),           \
                                                &detail::inline_##NAME<Src0TypeID, Src1TypeID>::f>(                    \
                ndt::make_type<decltype(dynd::nd::detail::inline_##NAME<Src0TypeID, Src1TypeID>::f)>())                \
      {                                                                                                                \
      }                                                                                                                \
    };                                                                                                                 \
  } /* namespace detail */                                                                                             \
  template <type_id_t Src0TypeID, type_id_t Src1TypeID>                                                                \
  using NAME##_callable = detail::NAME##_callable<Src0TypeID, Src1TypeID>;

  DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT(/, divide)
  DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT(%, mod)

#undef DYND_DEF_BINARY_OP_CALLABLE_ZEROCHECK_INT

  template <typename FuncType, bool Src0IsOption, bool Src1IsOption>
  struct option_arithmetic_kernel;

  template <typename FuncType>
  struct option_arithmetic_kernel<FuncType, true, false>
      : base_strided_kernel<option_arithmetic_kernel<FuncType, true, false>, 2> {
    intptr_t arith_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_na = this->get_child();
      bool1 child_dst;
      is_na->single(reinterpret_cast<char *>(&child_dst), &src[0]);
      if (!child_dst) {
        this->get_child(arith_offset)->single(dst, src);
      }
      else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }
  };

  template <typename FuncType>
  struct option_arithmetic_kernel<FuncType, false, true>
      : base_strided_kernel<option_arithmetic_kernel<FuncType, false, true>, 2> {
    intptr_t arith_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_na = this->get_child();
      bool1 child_dst;
      is_na->single(reinterpret_cast<char *>(&child_dst), &src[1]);
      if (!child_dst) {
        this->get_child(arith_offset)->single(dst, src);
      }
      else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }
  };

  template <typename FuncType>
  struct option_arithmetic_kernel<FuncType, true, true>
      : base_strided_kernel<option_arithmetic_kernel<FuncType, true, true>, 2> {
    intptr_t is_na_rhs_offset;
    intptr_t arith_offset;
    intptr_t assign_na_offset;

    void single(char *dst, char *const *src)
    {
      auto is_na_lhs = this->get_child();
      auto is_na_rhs = this->get_child(is_na_rhs_offset);
      bool child_dst_lhs;
      bool child_dst_rhs;
      is_na_lhs->single(reinterpret_cast<char *>(&child_dst_lhs), &src[0]);
      is_na_rhs->single(reinterpret_cast<char *>(&child_dst_rhs), &src[1]);
      if (!child_dst_lhs && !child_dst_rhs) {
        this->get_child(arith_offset)->single(dst, src);
      }
      else {
        this->get_child(assign_na_offset)->single(dst, nullptr);
      }
    }
  };

} // namespace dynd::nd

namespace ndt {

  DYND_ALLOW_UNSIGNED_UNARY_MINUS
  DYND_END_ALLOW_UNSIGNED_UNARY_MINUS

} // namespace dynd::ndt
} // namespace dynd
