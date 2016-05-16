//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/comparison_dispatch_callable.hpp>
#include <dynd/callables/equal_callable.hpp>
#include <dynd/callables/greater_callable.hpp>
#include <dynd/callables/greater_equal_callable.hpp>
#include <dynd/callables/less_callable.hpp>
#include <dynd/callables/less_equal_callable.hpp>
#include <dynd/callables/not_equal_callable.hpp>
#include <dynd/callables/total_order_callable.hpp>
#include <dynd/comparison.hpp>
#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

namespace {

template <template <typename...> class KernelType>
dispatcher<2, nd::callable> make_comparison_children() {
  typedef type_sequence<bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double>
      numeric_types;
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, uint8_id, uint16_id, uint32_id, uint64_id,
                           float32_id, float64_id>
      numeric_ids;

  dispatcher<2, nd::callable> dispatcher = nd::callable::make_all<KernelType, numeric_types, numeric_types>();

  for (type_id_t i0 : i2a<numeric_ids>()) {
    for (type_id_t i1 : i2a<dim_ids>()) {
      dispatcher.insert({{i0, i1}, nd::get_elwise()});
    }
  }

  for (type_id_t i : i2a<numeric_ids>()) {
    dispatcher.insert({{option_id, i}, nd::functional::forward_na<0>(ndt::type("Any"))});
    dispatcher.insert({{i, option_id}, nd::functional::forward_na<1>(ndt::type("Any"))});
  }
  dispatcher.insert({{option_id, option_id}, nd::functional::forward_na<0, 1>(ndt::type("Any"))});

  for (type_id_t dim_tp_id : i2a<dim_ids>()) {
    dispatcher.insert({{dim_tp_id, option_id}, nd::get_elwise()});
    dispatcher.insert({{option_id, dim_tp_id}, nd::get_elwise()});
  }

  for (type_id_t i0 : i2a<dim_ids>()) {
    typedef join<numeric_ids, dim_ids>::type type_ids;
    for (type_id_t i1 : i2a<type_ids>()) {
      dispatcher.insert({{i0, i1}, nd::get_elwise()});
    }
  }

  dispatcher.insert({{string_id, string_id}, nd::make_callable<KernelType<dynd::string, dynd::string>>()});

  return dispatcher;
}

template <template <typename...> class CallableType>
nd::callable make_comparison_callable() {
  return nd::make_callable<nd::comparison_dispatch_callable>(ndt::type("(Any, Any) -> Any"),
                                                             make_comparison_children<CallableType>());
}

nd::callable make_less() { return make_comparison_callable<nd::less_callable>(); }

nd::callable make_less_equal() { return make_comparison_callable<nd::less_equal_callable>(); }

nd::callable make_equal() {
  dispatcher<2, nd::callable> dispatcher = make_comparison_children<nd::equal_callable>();
  dispatcher.insert({{complex_float32_id, complex_float32_id},
                     nd::make_callable<nd::equal_callable<dynd::complex<float>, dynd::complex<float>>>()});
  dispatcher.insert({{complex_float64_id, complex_float64_id},
                     nd::make_callable<nd::equal_callable<dynd::complex<double>, dynd::complex<double>>>()});
  dispatcher.insert({{tuple_id, tuple_id}, nd::make_callable<nd::equal_callable<ndt::tuple_type, ndt::tuple_type>>()});
  dispatcher.insert(
      {{struct_id, struct_id}, nd::make_callable<nd::equal_callable<ndt::tuple_type, ndt::tuple_type>>()});
  dispatcher.insert({{type_id, type_id}, nd::make_callable<nd::equal_callable<ndt::type, ndt::type>>()});
  dispatcher.insert({{bytes_id, bytes_id}, nd::make_callable<nd::equal_callable<bytes, bytes>>()});

  return nd::make_callable<nd::comparison_dispatch_callable>(ndt::type("(Any, Any) -> Any"), dispatcher);
}

nd::callable make_not_equal() {
  dispatcher<2, nd::callable> dispatcher = make_comparison_children<nd::not_equal_callable>();
  dispatcher.insert({{complex_float32_id, complex_float32_id},
                     nd::make_callable<nd::not_equal_callable<dynd::complex<float>, dynd::complex<float>>>()});
  dispatcher.insert({{complex_float64_id, complex_float64_id},
                     nd::make_callable<nd::not_equal_callable<dynd::complex<double>, dynd::complex<double>>>()});
  dispatcher.insert(
      {{tuple_id, tuple_id}, nd::make_callable<nd::not_equal_callable<ndt::tuple_type, ndt::tuple_type>>()});
  dispatcher.insert(
      {{struct_id, struct_id}, nd::make_callable<nd::not_equal_callable<ndt::tuple_type, ndt::tuple_type>>()});
  dispatcher.insert({{type_id, type_id}, nd::make_callable<nd::not_equal_callable<ndt::type, ndt::type>>()});
  dispatcher.insert({{bytes_id, bytes_id}, nd::make_callable<nd::not_equal_callable<bytes, bytes>>()});

  return nd::make_callable<nd::comparison_dispatch_callable>(ndt::type("(Any, Any) -> Any"), dispatcher);
}

nd::callable make_greater_equal() { return make_comparison_callable<nd::greater_equal_callable>(); }

nd::callable make_greater() { return make_comparison_callable<nd::greater_callable>(); }

nd::callable make_total_order() {
  return nd::make_callable<nd::comparison_dispatch_callable>(
      ndt::type("(Any, Any) -> Any"),
      dispatcher<2, nd::callable>{
          //          {{fixed_string_id, fixed_string_id},
          //         nd::make_callable<nd::total_order_callable<fixed_string_id, fixed_string_id>>()},
          {{string_id, string_id}, nd::make_callable<nd::total_order_callable<dynd::string, dynd::string>>()},
          {{int32_id, int32_id}, nd::make_callable<nd::total_order_callable<int32_t, int32_t>>()},
          {{bool_id, bool_id}, nd::make_callable<nd::total_order_callable<bool, bool>>()}});
}

} // unnamed namespace

DYND_API nd::callable nd::less = make_less();
DYND_API nd::callable nd::less_equal = make_less_equal();
DYND_API nd::callable nd::equal = make_equal();
DYND_API nd::callable nd::not_equal = make_not_equal();
DYND_API nd::callable nd::greater_equal = make_greater_equal();
DYND_API nd::callable nd::greater = make_greater();
DYND_API nd::callable nd::total_order = make_total_order();
