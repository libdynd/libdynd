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

  static const std::vector<ndt::type> numeric_dyn_types = {
      ndt::make_type<bool>(),     ndt::make_type<int8_t>(),  ndt::make_type<int16_t>(),  ndt::make_type<int32_t>(),
      ndt::make_type<int64_t>(),  ndt::make_type<uint8_t>(), ndt::make_type<uint16_t>(), ndt::make_type<uint32_t>(),
      ndt::make_type<uint64_t>(), ndt::make_type<float>(),   ndt::make_type<double>()};

  dispatcher<2, nd::callable> dispatcher = nd::callable::make_all<KernelType, numeric_types, numeric_types>();

  for (auto i0 : numeric_dyn_types) {
    for (auto i1 : {ndt::make_type<ndt::fixed_dim_kind_type>(), ndt::make_type<ndt::var_dim_type>()}) {
      dispatcher.insert({{i0, i1}, nd::get_elwise()});
    }
  }

  for (auto i : numeric_dyn_types) {
    dispatcher.insert({{ndt::make_type<ndt::option_type>(), i}, nd::functional::forward_na<0>(ndt::type("Any"))});
    dispatcher.insert({{i, ndt::make_type<ndt::option_type>()}, nd::functional::forward_na<1>(ndt::type("Any"))});
  }
  dispatcher.insert({{ndt::make_type<ndt::option_type>(), ndt::make_type<ndt::option_type>()},
                     nd::functional::forward_na<0, 1>(ndt::type("Any"))});

  for (auto dim_tp_id : {ndt::make_type<ndt::fixed_dim_kind_type>(), ndt::make_type<ndt::var_dim_type>()}) {
    dispatcher.insert({{dim_tp_id, ndt::make_type<ndt::option_type>()}, nd::get_elwise()});
    dispatcher.insert({{ndt::make_type<ndt::option_type>(), dim_tp_id}, nd::get_elwise()});
  }

  for (auto i0 : {ndt::make_type<ndt::fixed_dim_kind_type>(), ndt::make_type<ndt::var_dim_type>()}) {
    for (auto i1 : {ndt::make_type<ndt::fixed_dim_kind_type>(), ndt::make_type<ndt::var_dim_type>()}) {
      dispatcher.insert({{i0, i1}, nd::get_elwise()});
    }
    for (auto i1 : numeric_dyn_types) {
      dispatcher.insert({{i0, i1}, nd::get_elwise()});
    }
  }

  dispatcher.insert({{ndt::make_type<dynd::string>(), ndt::make_type<dynd::string>()},
                     nd::make_callable<KernelType<dynd::string, dynd::string>>()});

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
  dispatcher.insert({{ndt::make_type<dynd::complex<float>>(), ndt::make_type<dynd::complex<float>>()},
                     nd::make_callable<nd::equal_callable<dynd::complex<float>, dynd::complex<float>>>()});
  dispatcher.insert({{ndt::make_type<dynd::complex<double>>(), ndt::make_type<dynd::complex<double>>()},
                     nd::make_callable<nd::equal_callable<dynd::complex<double>, dynd::complex<double>>>()});
  dispatcher.insert({{ndt::make_type<ndt::tuple_type>(), ndt::make_type<ndt::tuple_type>()},
                     nd::make_callable<nd::equal_callable<ndt::tuple_type, ndt::tuple_type>>()});
  dispatcher.insert({{ndt::make_type<ndt::struct_type>(), ndt::make_type<ndt::struct_type>()},
                     nd::make_callable<nd::equal_callable<ndt::struct_type, ndt::struct_type>>()});
  dispatcher.insert({{ndt::make_type<ndt::type>(), ndt::make_type<ndt::type>()},
                     nd::make_callable<nd::equal_callable<ndt::type, ndt::type>>()});
  dispatcher.insert({{ndt::make_type<ndt::bytes_type>(), ndt::make_type<ndt::bytes_type>()},
                     nd::make_callable<nd::equal_callable<bytes, bytes>>()});

  return nd::make_callable<nd::comparison_dispatch_callable>(ndt::type("(Any, Any) -> Any"), dispatcher);
}

nd::callable make_not_equal() {
  dispatcher<2, nd::callable> dispatcher = make_comparison_children<nd::not_equal_callable>();
  dispatcher.insert({{ndt::make_type<dynd::complex<float>>(), ndt::make_type<dynd::complex<float>>()},
                     nd::make_callable<nd::not_equal_callable<dynd::complex<float>, dynd::complex<float>>>()});
  dispatcher.insert({{ndt::make_type<dynd::complex<double>>(), ndt::make_type<dynd::complex<double>>()},
                     nd::make_callable<nd::not_equal_callable<dynd::complex<double>, dynd::complex<double>>>()});
  dispatcher.insert({{ndt::make_type<ndt::tuple_type>(), ndt::make_type<ndt::tuple_type>()},
                     nd::make_callable<nd::not_equal_callable<ndt::tuple_type, ndt::tuple_type>>()});
  dispatcher.insert({{ndt::make_type<ndt::struct_type>(), ndt::make_type<ndt::struct_type>()},
                     nd::make_callable<nd::not_equal_callable<ndt::struct_type, ndt::struct_type>>()});
  dispatcher.insert({{ndt::make_type<ndt::type>(), ndt::make_type<ndt::type>()},
                     nd::make_callable<nd::not_equal_callable<ndt::type, ndt::type>>()});
  dispatcher.insert({{ndt::make_type<ndt::bytes_type>(), ndt::make_type<ndt::bytes_type>()},
                     nd::make_callable<nd::not_equal_callable<bytes, bytes>>()});

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
          {{ndt::make_type<dynd::string>(), ndt::make_type<dynd::string>()},
           nd::make_callable<nd::total_order_callable<dynd::string, dynd::string>>()},
          {{ndt::make_type<int32_t>(), ndt::make_type<int32_t>()},
           nd::make_callable<nd::total_order_callable<int32_t, int32_t>>()},
          {{ndt::make_type<bool>(), ndt::make_type<bool>()},
           nd::make_callable<nd::total_order_callable<bool, bool>>()}});
}

} // unnamed namespace

DYND_API nd::callable nd::less = make_less();
DYND_API nd::callable nd::less_equal = make_less_equal();
DYND_API nd::callable nd::equal = make_equal();
DYND_API nd::callable nd::not_equal = make_not_equal();
DYND_API nd::callable nd::greater_equal = make_greater_equal();
DYND_API nd::callable nd::greater = make_greater();
DYND_API nd::callable nd::total_order = make_total_order();
