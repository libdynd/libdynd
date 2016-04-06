//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arithmetic.hpp>
#include <dynd/functional.hpp>
#include <dynd/callables/arithmetic_dispatch_callable.hpp>
#include <dynd/callables/compound_arithmetic_dispatch_callable.hpp>
#include <dynd/callables/compound_add_callable.hpp>
#include <dynd/callables/compound_div_callable.hpp>
#include <dynd/callables/option_arithmetic_callable.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/callables/sum_dispatch_callable.hpp>
#include <dynd/callables/sum_callable.hpp>

using namespace std;
using namespace dynd;

namespace {

typedef type_id_sequence<uint8_id, uint16_id, uint32_id, uint64_id, int8_id, int16_id, int32_id, int64_id, float32_id,
                         float64_id, complex_float32_id, complex_float64_id> binop_ids;

typedef type_id_sequence<uint8_id, uint16_id, uint32_id, uint64_id, int8_id, int16_id, int32_id, int64_id, float32_id,
                         float64_id> binop_real_ids;

template <nd::callable &Callable, template <type_id_t> class CallableType, typename TypeIDSequence>
nd::callable make_unary_arithmetic() {
  dispatcher<nd::callable> dispatcher = nd::callable::new_make_all<CallableType, TypeIDSequence>();

  const nd::callable self = nd::functional::call<Callable>(ndt::type("(Any) -> Any"));

  for (type_id_t i0 : i2a<dim_ids>()) {
    const ndt::type child_tp = ndt::callable_type::make(self.get_type()->get_return_type(), ndt::type(i0));
    dispatcher.insert({{i0}, nd::functional::cyclic_elwise(child_tp)});
  }

  return nd::make_callable<nd::arithmetic_dispatch_callable<1>>(self.get_array_type(), dispatcher);
}

template <nd::callable &Callable, template <type_id_t, type_id_t> class KernelType, typename TypeIDSequence>
nd::callable make_binary_arithmetic() {
  nd::callable self = nd::functional::call<Callable>(ndt::type("(Any, Any) -> Any"));

  auto dispatcher = nd::callable::new_make_all<KernelType, TypeIDSequence, TypeIDSequence>();
  dispatcher.insert(
      {{{option_id, any_kind_id}, nd::make_callable<nd::option_arithmetic_callable<Callable, true, false>>()},
       {{any_kind_id, option_id}, nd::make_callable<nd::option_arithmetic_callable<Callable, false, true>>()},
       {{option_id, option_id}, nd::make_callable<nd::option_arithmetic_callable<Callable, true, true>>()},
       {{dim_kind_id, scalar_kind_id}, nd::functional::cyclic_elwise(ndt::type("(Any, Any) -> Any"))},
       {{scalar_kind_id, dim_kind_id}, nd::functional::cyclic_elwise(ndt::type("(Any, Any) -> Any"))},
       {{dim_kind_id, dim_kind_id}, nd::functional::cyclic_elwise(ndt::type("(Any, Any) -> Any"))}});

  return nd::make_callable<nd::arithmetic_dispatch_callable<2>>(ndt::type("(Any, Any) -> Any"), dispatcher);
}

template <nd::callable &Callable, template <type_id_t, type_id_t> class KernelType, typename TypeIDSequence>
nd::callable make_compound_arithmetic() {
  auto dispatcher = nd::callable::new_make_all<KernelType, TypeIDSequence, TypeIDSequence>();

  nd::callable self = nd::functional::call<Callable>(ndt::type("(Any, Any) -> Any"));
  for (type_id_t i0 : i2a<TypeIDSequence>()) {
    for (type_id_t i1 : i2a<dim_ids>()) {
      dispatcher.insert({{i0, i1}, nd::functional::elwise(self)});
    }
  }

  for (type_id_t i0 : i2a<dim_ids>()) {
    typedef typename join<TypeIDSequence, dim_ids>::type broadcast_ids;
    for (type_id_t i1 : i2a<broadcast_ids>()) {
      dispatcher.insert({{i0, i1}, nd::functional::elwise(self)});
    }
  }

  return nd::make_callable<nd::compound_arithmetic_dispatch_callable>(ndt::type("(Any) -> Any"), dispatcher);
}

} // unnamed namespace

DYND_API nd::callable nd::plus = make_unary_arithmetic<nd::plus, nd::plus_callable, arithmetic_ids>();
DYND_API nd::callable nd::minus = make_unary_arithmetic<nd::minus, nd::minus_callable, arithmetic_ids>();
DYND_API nd::callable nd::logical_not =
    make_unary_arithmetic<nd::logical_not, nd::logical_not_callable, arithmetic_ids>();
DYND_API nd::callable nd::bitwise_not =
    make_unary_arithmetic<nd::bitwise_not, nd::bitwise_not_callable, integral_ids>();

DYND_API nd::callable nd::add = make_binary_arithmetic<nd::add, nd::add_callable, binop_ids>();
DYND_API nd::callable nd::subtract = make_binary_arithmetic<nd::subtract, nd::subtract_callable, binop_ids>();
DYND_API nd::callable nd::multiply = make_binary_arithmetic<nd::multiply, nd::multiply_callable, binop_ids>();
DYND_API nd::callable nd::divide = make_binary_arithmetic<nd::divide, nd::divide_callable, binop_ids>();
DYND_API nd::callable nd::logical_and =
    make_binary_arithmetic<nd::logical_and, nd::logical_and_callable, binop_real_ids>();
DYND_API nd::callable nd::logical_or =
    make_binary_arithmetic<nd::logical_or, nd::logical_or_callable, binop_real_ids>();

DYND_API nd::callable nd::compound_add =
    make_compound_arithmetic<nd::compound_add, nd::compound_add_callable, binop_ids>();
DYND_API nd::callable nd::compound_div =
    make_compound_arithmetic<nd::compound_div, nd::compound_div_callable, binop_ids>();

DYND_API nd::callable nd::sum = nd::functional::reduction(nd::make_callable<nd::sum_dispatch_callable>(
    ndt::callable_type::make(ndt::scalar_kind_type::make(), ndt::scalar_kind_type::make()),
    nd::callable::new_make_all<
        nd::sum_callable,
        type_id_sequence<int8_id, int16_id, int32_id, int64_id, uint8_id, uint16_id, uint32_id, uint64_id, float16_id,
                         float32_id, float64_id, complex_float32_id, complex_float64_id>>()));
