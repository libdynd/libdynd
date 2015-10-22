//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/callable.hpp>
#include <dynd/func/call.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/kernels/arithmetic.hpp>
#include <dynd/kernels/compound_add_kernel.hpp>
#include <dynd/kernels/compound_div_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename F, template <type_id_t...> class K, int N, typename TypeIDSequence>
  struct arithmetic_operator;

  template <typename FuncType, template <type_id_t> class KernelType, typename TypeIDSequence>
  struct arithmetic_operator<FuncType, KernelType, 1, TypeIDSequence> : declfunc<FuncType> {
    static std::map<type_id_t, callable> children;

    static callable &overload(const ndt::type &src0_tp)
    {
      return children[src0_tp.get_type_id()];
    }

    static callable make()
    {
      children = callable::make_all<KernelType, TypeIDSequence>(0);

      const callable self = functional::call<FuncType>(ndt::type("(Any) -> Any"));

      for (type_id_t i0 : dim_type_ids()) {
        const ndt::type child_tp = ndt::callable_type::make(self.get_type()->get_return_type(), ndt::type(i0));
        children[i0] = functional::elwise(child_tp, self);
      }

      return functional::multidispatch(
          self.get_array_type(),
          [](const ndt::type & DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc), const ndt::type * src_tp)->callable & {
            callable &child = overload(src_tp[0]);
            if (child.is_null()) {
              throw std::runtime_error(FuncType::what(src_tp[0]));
            }

            return child;
          },
          0);
    }
  };

  template <typename FuncType, template <type_id_t> class KernelType, typename TypeIDSequence>
  std::map<type_id_t, callable> arithmetic_operator<FuncType, KernelType, 1, TypeIDSequence>::children;

#define DYND_DEF_UNARY_OP_CALLABLE(NAME, TYPES)                                                                        \
  extern DYND_API struct NAME : arithmetic_operator<NAME, NAME##_kernel, 1, TYPES> {                                   \
    static std::string what(const ndt::type &src0_type)                                                                \
    {                                                                                                                  \
      std::stringstream ss;                                                                                            \
      ss << "no viable overload for dynd::nd::" #NAME " with argument type \"" << src0_type << "\"";                   \
      return ss.str();                                                                                                 \
    }                                                                                                                  \
  } NAME;

  DYND_DEF_UNARY_OP_CALLABLE(plus, arithmetic_type_ids)
  DYND_DEF_UNARY_OP_CALLABLE(minus, arithmetic_type_ids)
  DYND_DEF_UNARY_OP_CALLABLE(logical_not, arithmetic_type_ids)
  DYND_DEF_UNARY_OP_CALLABLE(bitwise_not, integral_type_ids)

#undef DYND_DEF_UNARY_OP_CALLABLE

  template <typename FuncType, template <type_id_t, type_id_t> class KernelType, typename TypeIDSequence>
  struct arithmetic_operator<FuncType, KernelType, 2, TypeIDSequence> : declfunc<FuncType> {
    static std::map<std::array<type_id_t, 2>, callable> children;

    static callable &overload(const ndt::type &src0_tp, const ndt::type &src1_tp)
    {
      return children[{{src0_tp.get_type_id(), src1_tp.get_type_id()}}];
    }

    static void fill()
    {
      children = callable::make_all<KernelType, TypeIDSequence, TypeIDSequence>(0);

      for (type_id_t i : TypeIDSequence()) {
        children[{{option_type_id, i}}] = callable::make<option_arithmetic_kernel<FuncType, true, false>>();
        children[{{i, option_type_id}}] = callable::make<option_arithmetic_kernel<FuncType, false, true>>();
      }
      children[{{option_type_id, option_type_id}}] = callable::make<option_arithmetic_kernel<FuncType, true, true>>();

      callable self = functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));
      for (type_id_t i0 : TypeIDSequence()) {
        for (type_id_t i1 : dim_type_ids()) {
          children[{{i0, i1}}] = functional::elwise(self);
        }
      }

      for (type_id_t i0 : dim_type_ids()) {
        typedef typename join<TypeIDSequence, dim_type_ids>::type broadcast_type_ids;
        for (type_id_t i1 : broadcast_type_ids()) {
          children[{{i0, i1}}] = functional::elwise(self);
        }
      }
    }

    static callable make()
    {
      FuncType::fill();

      return functional::multidispatch(
          ndt::type("(Any, Any) -> Any"),
          [](const ndt::type & DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc), const ndt::type * src_tp)->callable & {
            callable &child = overload(src_tp[0], src_tp[1]);
            if (child.is_null()) {
              throw std::runtime_error(FuncType::what(src_tp[0], src_tp[1]));
            }

            return child;
          },
          0);
    }
  };

  template <typename FuncType, template <type_id_t, type_id_t> class KernelType, typename TypeIDSequence>
  std::map<std::array<type_id_t, 2>, callable> arithmetic_operator<FuncType, KernelType, 2, TypeIDSequence>::children;

#define DYND_DEF_BINARY_OP_CALLABLE(NAME, TYPES)                                                                       \
  extern DYND_API struct NAME : arithmetic_operator<NAME, NAME##_kernel, 2, TYPES> {                                   \
    static std::string what(const ndt::type &src0_tp, const ndt::type &src1_tp)                                        \
    {                                                                                                                  \
      std::stringstream ss;                                                                                            \
      ss << "no viable overload for dynd::nd::" #NAME " with argument types \"" << src0_tp << "\" and \"" << src1_tp   \
         << "\"";                                                                                                      \
      return ss.str();                                                                                                 \
    }                                                                                                                  \
  } NAME;

  namespace detail {

    typedef type_id_sequence<uint8_type_id, uint16_type_id, uint32_type_id, uint64_type_id, int8_type_id,
                             int16_type_id, int32_type_id, int64_type_id, float32_type_id, float64_type_id,
                             complex_float32_type_id, complex_float64_type_id> binop_type_ids;

    typedef type_id_sequence<uint8_type_id, uint16_type_id, uint32_type_id, uint64_type_id, int8_type_id,
                             int16_type_id, int32_type_id, int64_type_id, float32_type_id,
                             float64_type_id> binop_real_type_ids;

  }

  DYND_DEF_BINARY_OP_CALLABLE(add, detail::binop_type_ids)
  DYND_DEF_BINARY_OP_CALLABLE(subtract, detail::binop_type_ids)
  DYND_DEF_BINARY_OP_CALLABLE(multiply, detail::binop_type_ids)
  DYND_DEF_BINARY_OP_CALLABLE(divide, detail::binop_type_ids)
  DYND_DEF_BINARY_OP_CALLABLE(logical_and, detail::binop_real_type_ids)
  DYND_DEF_BINARY_OP_CALLABLE(logical_or, detail::binop_real_type_ids)

#undef DYND_DEF_BINARY_OP_CALLABLE

  template <typename FuncType, template <type_id_t, type_id_t> class KernelType, typename TypeIDSequence>
  struct compound_arithmetic_operator : declfunc<FuncType> {
    static std::map<std::array<type_id_t, 2>, callable> children;

    static void fill()
    {
      children = callable::make_all<KernelType, TypeIDSequence, TypeIDSequence>();

      callable self = functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));
      for (type_id_t i0 : TypeIDSequence()) {
        for (type_id_t i1 : dim_type_ids()) {
          children[{{i0, i1}}] = functional::elwise(self);
        }
      }

      for (type_id_t i0 : dim_type_ids()) {
        typedef typename join<TypeIDSequence, dim_type_ids>::type broadcast_type_ids;
        for (type_id_t i1 : broadcast_type_ids()) {
          children[{{i0, i1}}] = functional::elwise(self);
        }
      }
    }

    static callable make()
    {
      FuncType::fill();

      return functional::multidispatch(
          ndt::type("(Any) -> Any"),
          [](const ndt::type & dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type * src_tp)->callable & {
            callable &child = children[{{dst_tp.get_type_id(), src_tp[0].get_type_id()}}];
            if (child.is_null()) {
              throw std::runtime_error("no child found");
            }

            return child;
          },
          0);
    }
  };

  template <typename FuncType, template <type_id_t, type_id_t> class KernelType, typename TypeIDSequence>
  std::map<std::array<type_id_t, 2>, callable>
  compound_arithmetic_operator<FuncType, KernelType, TypeIDSequence>::children;

#define DYND_DEF_COMPOUND_OP_CALLABLE(NAME, TYPES)                                                                     \
  extern DYND_API struct NAME : compound_arithmetic_operator<NAME, NAME##_kernel_t, TYPES> {                           \
  } NAME;

  DYND_DEF_COMPOUND_OP_CALLABLE(compound_add, detail::binop_type_ids)
  DYND_DEF_COMPOUND_OP_CALLABLE(compound_div, detail::binop_type_ids)

#undef DYND_DEF_COMPOUND_OP_CALLABLE

} // namespace dynd::nd
} // namespace dynd
