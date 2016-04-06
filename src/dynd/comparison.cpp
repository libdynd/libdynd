//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/comparison.hpp>
#include <dynd/functional.hpp>
#include <dynd/callables/comparison_dispatch_callable.hpp>
#include <dynd/callables/equal_callable.hpp>
#include <dynd/callables/greater_callable.hpp>
#include <dynd/callables/greater_equal_callable.hpp>
#include <dynd/callables/less_callable.hpp>
#include <dynd/callables/less_equal_callable.hpp>
#include <dynd/callables/not_equal_callable.hpp>
#include <dynd/callables/total_order_callable.hpp>

using namespace std;
using namespace dynd;

namespace {

template <nd::callable &Func, template <type_id_t...> class KernelType>
dispatcher<nd::callable> make_comparison_children() {
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, uint8_id, uint16_id, uint32_id, uint64_id,
                           float32_id, float64_id> numeric_ids;

  dispatcher<nd::callable> dispatcher = nd::callable::new_make_all<KernelType, numeric_ids, numeric_ids>();

  for (type_id_t i0 : i2a<numeric_ids>()) {
    for (type_id_t i1 : i2a<dim_ids>()) {
      const ndt::type child_tp = ndt::callable_type::make(ndt::type("Any"), {ndt::type(i0), ndt::type(i1)});
      dispatcher.insert({{i0, i1}, nd::functional::elwise(child_tp)});
    }
  }

  for (type_id_t i : i2a<numeric_ids>()) {
    dispatcher.insert({{option_id, i}, nd::functional::forward_na<0>(ndt::type("Any"))});
    dispatcher.insert({{i, option_id}, nd::functional::forward_na<1>(ndt::type("Any"))});
  }
  dispatcher.insert({{option_id, option_id}, nd::make_callable<nd::option_comparison_callable<true, true>>()});

  for (type_id_t dim_tp_id : i2a<dim_ids>()) {
    dispatcher.insert({{dim_tp_id, option_id}, nd::functional::elwise(ndt::type("(Any, Any) -> Any"))});
    dispatcher.insert({{option_id, dim_tp_id}, nd::functional::elwise(ndt::type("(Any, Any) -> Any"))});
  }

  for (type_id_t i0 : i2a<dim_ids>()) {
    typedef join<numeric_ids, dim_ids>::type type_ids;
    for (type_id_t i1 : i2a<type_ids>()) {
      const ndt::type child_tp = ndt::callable_type::make(ndt::type("Any"), {ndt::type(i0), ndt::type(i1)});
      dispatcher.insert({{i0, i1}, nd::functional::elwise(child_tp)});
    }
  }

  dispatcher.insert({{string_id, string_id}, nd::make_callable<KernelType<string_id, string_id>>()});

  return dispatcher;
}

template <nd::callable &Callable, template <type_id_t...> class CallableType>
nd::callable make_comparison_callable() {
  return nd::make_callable<nd::comparison_dispatch_callable>(ndt::type("(Any, Any) -> Any"),
                                                             make_comparison_children<Callable, CallableType>());
}

nd::callable make_less() { return make_comparison_callable<nd::less, nd::less_callable>(); }

nd::callable make_less_equal() { return make_comparison_callable<nd::less_equal, nd::less_equal_callable>(); }

nd::callable make_equal() {
  dispatcher<nd::callable> dispatcher = make_comparison_children<nd::equal, nd::equal_callable>();
  dispatcher.insert({{complex_float32_id, complex_float32_id},
                     nd::make_callable<nd::equal_callable<complex_float32_id, complex_float32_id>>()});
  dispatcher.insert({{complex_float64_id, complex_float64_id},
                     nd::make_callable<nd::equal_callable<complex_float64_id, complex_float64_id>>()});
  dispatcher.insert({{tuple_id, tuple_id}, nd::make_callable<nd::equal_callable<tuple_id, tuple_id>>()});
  dispatcher.insert({{struct_id, struct_id}, nd::make_callable<nd::equal_callable<tuple_id, tuple_id>>()});
  dispatcher.insert({{type_id, type_id}, nd::make_callable<nd::equal_callable<type_id, type_id>>()});
  dispatcher.insert({{bytes_id, bytes_id}, nd::make_callable<nd::equal_callable<bytes_id, bytes_id>>()});

  return nd::make_callable<nd::comparison_dispatch_callable>(ndt::type("(Any, Any) -> Any"), dispatcher);
}

nd::callable make_not_equal() {
  dispatcher<nd::callable> dispatcher = make_comparison_children<nd::not_equal, nd::not_equal_callable>();
  dispatcher.insert({{complex_float32_id, complex_float32_id},
                     nd::make_callable<nd::not_equal_callable<complex_float32_id, complex_float32_id>>()});
  dispatcher.insert({{complex_float64_id, complex_float64_id},
                     nd::make_callable<nd::not_equal_callable<complex_float64_id, complex_float64_id>>()});
  dispatcher.insert({{tuple_id, tuple_id}, nd::make_callable<nd::not_equal_callable<tuple_id, tuple_id>>()});
  dispatcher.insert({{struct_id, struct_id}, nd::make_callable<nd::not_equal_callable<tuple_id, tuple_id>>()});
  dispatcher.insert({{type_id, type_id}, nd::make_callable<nd::not_equal_callable<type_id, type_id>>()});
  dispatcher.insert({{bytes_id, bytes_id}, nd::make_callable<nd::not_equal_callable<bytes_id, bytes_id>>()});

  return nd::make_callable<nd::comparison_dispatch_callable>(ndt::type("(Any, Any) -> Any"), dispatcher);
}

nd::callable make_greater_equal() { return make_comparison_callable<nd::greater_equal, nd::greater_equal_callable>(); }

nd::callable make_greater() { return make_comparison_callable<nd::greater, nd::greater_callable>(); }

nd::callable make_total_order() {
  return nd::make_callable<nd::comparison_dispatch_callable>(
      ndt::type("(Any, Any) -> Any"),
      dispatcher<nd::callable>{
          {{fixed_string_id, fixed_string_id},
           nd::make_callable<nd::total_order_callable<fixed_string_id, fixed_string_id>>()},
          {{string_id, string_id}, nd::make_callable<nd::total_order_callable<string_id, string_id>>()},
          {{int32_id, int32_id}, nd::make_callable<nd::total_order_callable<int32_id, int32_id>>()},
          {{bool_id, bool_id}, nd::make_callable<nd::total_order_callable<bool_id, bool_id>>()}});
}

} // unnamed namespace

DYND_API nd::callable nd::less = make_less();
DYND_API nd::callable nd::less_equal = make_less_equal();
DYND_API nd::callable nd::equal = make_equal();
DYND_API nd::callable nd::not_equal = make_not_equal();
DYND_API nd::callable nd::greater_equal = make_greater_equal();
DYND_API nd::callable nd::greater = make_greater();
DYND_API nd::callable nd::total_order = make_total_order();
