//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/callables/arithmetic_dispatch_callable.hpp>
#include <dynd/callables/compound_arithmetic_dispatch_callable.hpp>
#include <dynd/kernels/arithmetic.hpp>
#include <dynd/callables/compound_add_callable.hpp>
#include <dynd/callables/compound_div_callable.hpp>
#include <dynd/callables/option_arithmetic_callable.hpp>

namespace dynd {
namespace nd {

  template <typename FuncType, template <type_id_t> class CallableType, typename TypeIDSequence>
  struct unary_arithmetic_operator : declfunc<FuncType> {
    static callable make()
    {
      auto dispatcher = callable::new_make_all<CallableType, TypeIDSequence>();

      const callable self = functional::call<FuncType>(ndt::type("(Any) -> Any"));

      for (type_id_t i0 : i2a<dim_ids>()) {
        const ndt::type child_tp = ndt::callable_type::make(self.get_type()->get_return_type(), ndt::type(i0));
        dispatcher.insert({{i0}, functional::elwise(child_tp, self)});
      }

      return make_callable<arithmetic_dispatch_callable<1>>(self.get_array_type(), dispatcher);
    }
  };

#define DYND_DEF_UNARY_OP_CALLABLE(NAME, TYPES)                                                                        \
  extern DYND_API struct DYND_API NAME : unary_arithmetic_operator<NAME, NAME##_callable, TYPES> {                     \
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
      callable self = functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));

      auto dispatcher = callable::new_make_all<KernelType, TypeIDSequence, TypeIDSequence>();
      dispatcher.insert({{{option_id, any_kind_id}, make_callable<option_arithmetic_callable<FuncType, true, false>>()},
                         {{any_kind_id, option_id}, make_callable<option_arithmetic_callable<FuncType, false, true>>()},
                         {{option_id, option_id}, make_callable<option_arithmetic_callable<FuncType, true, true>>()},
                         {{dim_kind_id, scalar_kind_id}, functional::elwise(self)},
                         {{scalar_kind_id, dim_kind_id}, functional::elwise(self)},
                         {{dim_kind_id, dim_kind_id}, functional::elwise(self)}});

      return make_callable<arithmetic_dispatch_callable<2>>(ndt::type("(Any, Any) -> Any"), dispatcher);
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
                             float32_id, float64_id, complex_float32_id, complex_float64_id> binop_ids;

    typedef type_id_sequence<uint8_id, uint16_id, uint32_id, uint64_id, int8_id, int16_id, int32_id, int64_id,
                             float32_id, float64_id> binop_real_ids;
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
      auto dispatcher = callable::new_make_all<KernelType, TypeIDSequence, TypeIDSequence>();

      callable self = functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));
      for (type_id_t i0 : i2a<TypeIDSequence>()) {
        for (type_id_t i1 : i2a<dim_ids>()) {
          dispatcher.insert({{i0, i1}, functional::elwise(self)});
        }
      }

      for (type_id_t i0 : i2a<dim_ids>()) {
        typedef typename join<TypeIDSequence, dim_ids>::type broadcast_ids;
        for (type_id_t i1 : i2a<broadcast_ids>()) {
          dispatcher.insert({{i0, i1}, functional::elwise(self)});
        }
      }

      return make_callable<compound_arithmetic_dispatch_callable>(ndt::type("(Any) -> Any"), dispatcher);
    }
  };

#define DYND_DEF_COMPOUND_OP_CALLABLE(NAME, TYPES)                                                                     \
  extern DYND_API struct DYND_API NAME : compound_arithmetic_operator<NAME, NAME##_callable, TYPES> {                  \
    static nd::callable &get();                                                                                        \
  } NAME;

  DYND_DEF_COMPOUND_OP_CALLABLE(compound_add, detail::binop_ids)
  DYND_DEF_COMPOUND_OP_CALLABLE(compound_div, detail::binop_ids)

#undef DYND_DEF_COMPOUND_OP_CALLABLE

} // namespace dynd::nd
} // namespace dynd
