//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/all_equal_callable.hpp>
#include <dynd/callables/equal_callable.hpp>
#include <dynd/callables/greater_callable.hpp>
#include <dynd/callables/greater_equal_callable.hpp>
#include <dynd/callables/less_callable.hpp>
#include <dynd/callables/less_equal_callable.hpp>
#include <dynd/callables/multidispatch_callable.hpp>
#include <dynd/callables/not_equal_callable.hpp>
#include <dynd/callables/total_order_callable.hpp>
#include <dynd/comparison.hpp>
#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

namespace {

typedef type_sequence<bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double>
    numeric_types;

static std::vector<ndt::type> func_ptr(const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc),
                                       const ndt::type *src_tp) {
  return {src_tp[0], src_tp[1]};
}

template <std::vector<ndt::type> (*Func)(const ndt::type &, size_t, const ndt::type *),
          template <typename...> class KernelType>
dispatcher<2, nd::callable> make_comparison_children() {
  static const std::vector<ndt::type> numeric_dyn_types = {
      ndt::make_type<bool>(),     ndt::make_type<int8_t>(),  ndt::make_type<int16_t>(),  ndt::make_type<int32_t>(),
      ndt::make_type<int64_t>(),  ndt::make_type<uint8_t>(), ndt::make_type<uint16_t>(), ndt::make_type<uint32_t>(),
      ndt::make_type<uint64_t>(), ndt::make_type<float>(),   ndt::make_type<double>()};

  dispatcher<2, nd::callable> dispatcher = nd::callable::make_all<KernelType, numeric_types, numeric_types>(Func);
  dispatcher.insert({nd::get_elwise(ndt::type("(Dim, Scalar) -> Any")),
                     nd::get_elwise(ndt::type("(Scalar, Dim) -> Any")),
                     nd::get_elwise(ndt::type("(Dim, Dim) -> Any"))});

  dispatcher.insert({nd::functional::forward_na<0>(ndt::type("Any"), {ndt::type("?Any"), ndt::type("Any")}),
                     nd::functional::forward_na<1>(ndt::type("Any"), {ndt::type("Any"), ndt::type("?Any")}),
                     nd::functional::forward_na<0, 1>(ndt::type("Any"), {ndt::type("?Any"), ndt::type("?Any")})});

  dispatcher.insert(nd::get_elwise(ndt::type("(Dim, ?Any) -> Any")));
  dispatcher.insert(nd::get_elwise(ndt::type("(?Any, Dim) -> Any")));

  dispatcher.insert(nd::make_callable<KernelType<dynd::string, dynd::string>>());

  return dispatcher;
}

template <template <typename...> class CallableType>
nd::callable make_comparison_callable() {
  return nd::make_callable<nd::multidispatch_callable<2>>(ndt::type("(Any, Any) -> Any"),
                                                          make_comparison_children<func_ptr, CallableType>());
}

nd::callable make_less() { return make_comparison_callable<nd::less_callable>(); }

nd::callable make_less_equal() { return make_comparison_callable<nd::less_equal_callable>(); }

nd::callable make_equal() {
  dispatcher<2, nd::callable> dispatcher = make_comparison_children<func_ptr, nd::equal_callable>();
  dispatcher.insert(nd::make_callable<nd::equal_callable<dynd::complex<float>, dynd::complex<float>>>());
  dispatcher.insert(nd::make_callable<nd::equal_callable<dynd::complex<double>, dynd::complex<double>>>());
  dispatcher.insert(nd::make_callable<nd::equal_callable<ndt::tuple_type, ndt::tuple_type>>());
  dispatcher.insert(nd::make_callable<nd::equal_callable<ndt::struct_type, ndt::struct_type>>());
  dispatcher.insert(nd::make_callable<nd::equal_callable<ndt::type, ndt::type>>());
  dispatcher.insert(nd::make_callable<nd::equal_callable<bytes, bytes>>());

  return nd::make_callable<nd::multidispatch_callable<2>>(ndt::type("(Any, Any) -> Any"), dispatcher);
}

nd::callable make_not_equal() {
  dispatcher<2, nd::callable> dispatcher = make_comparison_children<func_ptr, nd::not_equal_callable>();
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<dynd::complex<float>, dynd::complex<float>>>());
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<dynd::complex<double>, dynd::complex<double>>>());
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<ndt::tuple_type, ndt::tuple_type>>());
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<ndt::struct_type, ndt::struct_type>>());
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<ndt::type, ndt::type>>());
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<bytes, bytes>>());

  return nd::make_callable<nd::multidispatch_callable<2>>(ndt::type("(Any, Any) -> Any"), dispatcher);
}

nd::callable make_greater_equal() { return make_comparison_callable<nd::greater_equal_callable>(); }

nd::callable make_greater() { return make_comparison_callable<nd::greater_callable>(); }

nd::callable make_total_order() {
  return nd::make_callable<nd::multidispatch_callable<2>>(
      ndt::type("(Any, Any) -> Any"),
      dispatcher<2, nd::callable>(func_ptr, {nd::make_callable<nd::total_order_callable<dynd::string, dynd::string>>(),
                                             nd::make_callable<nd::total_order_callable<int32_t, int32_t>>(),
                                             nd::make_callable<nd::total_order_callable<bool, bool>>()}));
}

nd::callable make_all_equal() {
  dispatcher<2, nd::callable> dispatcher =
      nd::callable::make_all<nd::all_equal_callable, numeric_types, numeric_types>(func_ptr);

  return nd::functional::reduction([] { return true; }, nd::make_callable<nd::multidispatch_callable<2>>(
                                                            ndt::type("(Scalar, Scalar) -> bool"), dispatcher));
}

} // unnamed namespace

DYND_API nd::callable nd::less = make_less();
DYND_API nd::callable nd::less_equal = make_less_equal();
DYND_API nd::callable nd::equal = make_equal();
DYND_API nd::callable nd::not_equal = make_not_equal();
DYND_API nd::callable nd::greater_equal = make_greater_equal();
DYND_API nd::callable nd::greater = make_greater();
DYND_API nd::callable nd::total_order = make_total_order();

DYND_API nd::callable nd::all_equal = make_all_equal();
