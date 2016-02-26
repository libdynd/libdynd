//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/kernels/arithmetic.hpp>
#include <dynd/kernels/compound_add_kernel.hpp>
#include <dynd/kernels/compound_div_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename FuncType, template <type_id_t> class KernelType, typename TypeIDSequence>
  struct unary_arithmetic_operator : declfunc<FuncType> {
    static callable make()
    {
      auto children = callable::make_all<KernelType, TypeIDSequence>();

      const callable self = functional::call<FuncType>(ndt::type("(Any) -> Any"));

      for (type_id_t i0 : i2a<dim_ids>()) {
        const ndt::type child_tp = ndt::callable_type::make(self.get_type()->get_return_type(), ndt::type(i0));
        children[i0] = functional::elwise(child_tp, self);
      }

      return functional::dispatch(self.get_array_type(),
                                  [children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                                             const ndt::type *src_tp) mutable -> callable & {
                                    callable &child = children[src_tp[0].get_id()];
                                    if (child.is_null()) {
                                      throw std::runtime_error(FuncType::what(src_tp[0]));
                                    }

                                    return child;
                                  });
    }
  };

#define DYND_DEF_UNARY_OP_CALLABLE(NAME, TYPES)                                                                        \
  extern DYND_API struct DYND_API NAME : unary_arithmetic_operator<NAME, NAME##_kernel, TYPES> {                       \
    static std::string what(const ndt::type &src0_type)                                                                \
    {                                                                                                                  \
      std::stringstream ss;                                                                                            \
      ss << "no viable overload for dynd::nd::" #NAME " with argument type \"" << src0_type << "\"";                   \
      return ss.str();                                                                                                 \
    }                                                                                                                  \
    static callable &get();                                                                                            \
  } NAME;

  DYND_DEF_UNARY_OP_CALLABLE(plus, arithmetic_ids)
  DYND_DEF_UNARY_OP_CALLABLE(minus, arithmetic_ids)
  DYND_DEF_UNARY_OP_CALLABLE(logical_not, arithmetic_ids)
  DYND_DEF_UNARY_OP_CALLABLE(bitwise_not, integral_ids)

#undef DYND_DEF_UNARY_OP_CALLABLE

  template <typename FuncType, template <type_id_t, type_id_t> class KernelType, typename TypeIDSequence>
  struct binary_arithmetic_operator : declfunc<FuncType> {
    static callable make()
    {
      auto children = callable::make_all<KernelType, TypeIDSequence, TypeIDSequence>();

      for (type_id_t i : i2a<TypeIDSequence>()) {
        children[{{option_id, i}}] = callable::make<option_arithmetic_kernel<FuncType, true, false>>();
        children[{{i, option_id}}] = callable::make<option_arithmetic_kernel<FuncType, false, true>>();
      }
      children[{{option_id, option_id}}] = callable::make<option_arithmetic_kernel<FuncType, true, true>>();

      callable self = functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));
      for (type_id_t i0 : i2a<TypeIDSequence>()) {
        for (type_id_t i1 : i2a<dim_ids>()) {
          children[{{i0, i1}}] = functional::elwise(self);
        }
      }

      for (type_id_t i0 : i2a<dim_ids>()) {
        typedef typename join<TypeIDSequence, dim_ids>::type broadcast_ids;
        for (type_id_t i1 : i2a<broadcast_ids>()) {
          children[{{i0, i1}}] = functional::elwise(self);
        }
      }

      return functional::dispatch(ndt::type("(Any, Any) -> Any"),
                                  [children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                                             const ndt::type *src_tp) mutable -> callable & {
                                    callable &child = children[{{src_tp[0].get_id(), src_tp[1].get_id()}}];
                                    if (child.is_null()) {
                                      throw std::runtime_error(FuncType::what(src_tp[0], src_tp[1]));
                                    }

                                    return child;
                                  });
    }
  };

#define DYND_DEF_BINARY_OP_CALLABLE(NAME, TYPES)                                                                       \
  extern DYND_API struct DYND_API NAME : binary_arithmetic_operator<NAME, NAME##_kernel, TYPES> {                      \
    static std::string what(const ndt::type &src0_tp, const ndt::type &src1_tp)                                        \
    {                                                                                                                  \
      std::stringstream ss;                                                                                            \
      ss << "no viable overload for dynd::nd::" #NAME " with argument types \"" << src0_tp << "\" and \"" << src1_tp   \
         << "\"";                                                                                                      \
      return ss.str();                                                                                                 \
    }                                                                                                                  \
    static nd::callable &get();                                                                                        \
  } NAME;

  namespace detail {

    typedef type_id_sequence<uint8_id, uint16_id, uint32_id, uint64_id, int8_id, int16_id, int32_id, int64_id,
                             float32_id, float64_id, complex_float32_id, complex_float64_id>
        binop_ids;

    typedef type_id_sequence<uint8_id, uint16_id, uint32_id, uint64_id, int8_id, int16_id, int32_id, int64_id,
                             float32_id, float64_id>
        binop_real_ids;
  }

  DYND_DEF_BINARY_OP_CALLABLE(add, detail::binop_ids)
  DYND_DEF_BINARY_OP_CALLABLE(subtract, detail::binop_ids)
  DYND_DEF_BINARY_OP_CALLABLE(multiply, detail::binop_ids)
  DYND_DEF_BINARY_OP_CALLABLE(divide, detail::binop_ids)
  DYND_DEF_BINARY_OP_CALLABLE(logical_and, detail::binop_real_ids)
  DYND_DEF_BINARY_OP_CALLABLE(logical_or, detail::binop_real_ids)

#undef DYND_DEF_BINARY_OP_CALLABLE

  template <typename FuncType, template <type_id_t, type_id_t> class KernelType, typename TypeIDSequence>
  struct compound_arithmetic_operator : declfunc<FuncType> {
    static callable make()
    {
      auto children = callable::make_all<KernelType, TypeIDSequence, TypeIDSequence>();

      callable self = functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));
      for (type_id_t i0 : i2a<TypeIDSequence>()) {
        for (type_id_t i1 : i2a<dim_ids>()) {
          children[{{i0, i1}}] = functional::elwise(self);
        }
      }

      for (type_id_t i0 : i2a<dim_ids>()) {
        typedef typename join<TypeIDSequence, dim_ids>::type broadcast_ids;
        for (type_id_t i1 : i2a<broadcast_ids>()) {
          children[{{i0, i1}}] = functional::elwise(self);
        }
      }

      return functional::dispatch(ndt::type("(Any) -> Any"),
                                  [children](const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                                             const ndt::type *src_tp) mutable -> callable & {
                                    callable &child = children[{{dst_tp.get_id(), src_tp[0].get_id()}}];
                                    if (child.is_null()) {
                                      throw std::runtime_error("no child found");
                                    }

                                    return child;
                                  });
    }
  };

#define DYND_DEF_COMPOUND_OP_CALLABLE(NAME, TYPES)                                                                     \
  extern DYND_API struct DYND_API NAME : compound_arithmetic_operator<NAME, NAME##_kernel_t, TYPES> {                  \
    static nd::callable &get();                                                                                        \
  } NAME;

  DYND_DEF_COMPOUND_OP_CALLABLE(compound_add, detail::binop_ids)
  DYND_DEF_COMPOUND_OP_CALLABLE(compound_div, detail::binop_ids)

#undef DYND_DEF_COMPOUND_OP_CALLABLE

} // namespace dynd::nd
} // namespace dynd
