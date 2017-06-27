//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This header is intended only to help with defining the various
// comparison callables and should not be used elsewhere.

//#include <dynd/callables/multidispatch_callable.hpp>

#include <dynd/callables/multidispatch_callable.hpp>
#include <dynd/comparison.hpp>
#include <dynd/functional.hpp>
#include <dynd/type_sequence.hpp>

namespace {

typedef dynd::type_sequence<bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float,
                            double>
    numeric_types;

static std::vector<dynd::ndt::type> func_ptr(const dynd::ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc),
                                             const dynd::ndt::type *src_tp) {
  return {src_tp[0], src_tp[1]};
}

template <std::vector<dynd::ndt::type> (*Func)(const dynd::ndt::type &, size_t, const dynd::ndt::type *),
          template <typename...> class KernelType>
dynd::dispatcher<2, dynd::nd::callable> make_comparison_children() {
  static const std::vector<dynd::ndt::type> numeric_dyn_types = {
      dynd::ndt::make_type<bool>(),     dynd::ndt::make_type<int8_t>(),   dynd::ndt::make_type<int16_t>(),
      dynd::ndt::make_type<int32_t>(),  dynd::ndt::make_type<int64_t>(),  dynd::ndt::make_type<uint8_t>(),
      dynd::ndt::make_type<uint16_t>(), dynd::ndt::make_type<uint32_t>(), dynd::ndt::make_type<uint64_t>(),
      dynd::ndt::make_type<float>(),    dynd::ndt::make_type<double>()};

  dynd::dispatcher<2, dynd::nd::callable> dispatcher =
      dynd::nd::callable::make_all<KernelType, numeric_types, numeric_types>(Func);
  dispatcher.insert(
      {dynd::nd::get_elwise(dynd::ndt::make_type<dynd::ndt::callable_type>(
           dynd::ndt::make_type<dynd::ndt::any_kind_type>(),
           {dynd::ndt::make_type<dynd::ndt::dim_kind_type>(dynd::ndt::make_type<dynd::ndt::any_kind_type>()),
            dynd::ndt::make_type<dynd::ndt::scalar_kind_type>()})),
       dynd::nd::get_elwise(dynd::ndt::make_type<dynd::ndt::callable_type>(
           dynd::ndt::make_type<dynd::ndt::any_kind_type>(),
           {dynd::ndt::make_type<dynd::ndt::scalar_kind_type>(),
            dynd::ndt::make_type<dynd::ndt::dim_kind_type>(dynd::ndt::make_type<dynd::ndt::any_kind_type>())})),
       dynd::nd::get_elwise(dynd::ndt::make_type<dynd::ndt::callable_type>(
           dynd::ndt::make_type<dynd::ndt::any_kind_type>(),
           {dynd::ndt::make_type<dynd::ndt::dim_kind_type>(dynd::ndt::make_type<dynd::ndt::any_kind_type>()),
            dynd::ndt::make_type<dynd::ndt::dim_kind_type>(dynd::ndt::make_type<dynd::ndt::any_kind_type>())}))});

  dispatcher.insert(
      {dynd::nd::functional::forward_na<0>(dynd::ndt::type("Any"), {dynd::ndt::type("?Any"), dynd::ndt::type("Any")}),
       dynd::nd::functional::forward_na<1>(dynd::ndt::type("Any"), {dynd::ndt::type("Any"), dynd::ndt::type("?Any")}),
       dynd::nd::functional::forward_na<0, 1>(dynd::ndt::type("Any"),
                                              {dynd::ndt::type("?Any"), dynd::ndt::type("?Any")})});

  dispatcher.insert(dynd::nd::get_elwise(dynd::ndt::make_type<dynd::ndt::callable_type>(
      dynd::ndt::make_type<dynd::ndt::any_kind_type>(),
      {dynd::ndt::make_type<dynd::ndt::dim_kind_type>(dynd::ndt::make_type<dynd::ndt::any_kind_type>()),
       dynd::ndt::make_type<dynd::ndt::option_type>(dynd::ndt::make_type<dynd::ndt::scalar_kind_type>())})));
  dispatcher.insert(dynd::nd::get_elwise(dynd::ndt::make_type<dynd::ndt::callable_type>(
      dynd::ndt::make_type<dynd::ndt::any_kind_type>(),
      {dynd::ndt::make_type<dynd::ndt::option_type>(dynd::ndt::make_type<dynd::ndt::scalar_kind_type>()),
       dynd::ndt::make_type<dynd::ndt::dim_kind_type>(dynd::ndt::make_type<dynd::ndt::any_kind_type>())})));

  dispatcher.insert(dynd::nd::make_callable<KernelType<dynd::string, dynd::string>>());

  return dispatcher;
}

template <template <typename...> class CallableType>
dynd::nd::callable make_comparison_callable() {
  return dynd::nd::make_callable<dynd::nd::multidispatch_callable<2>>(
      dynd::ndt::make_type<dynd::ndt::callable_type>(
          dynd::ndt::make_type<dynd::ndt::any_kind_type>(),
          {dynd::ndt::make_type<dynd::ndt::any_kind_type>(), dynd::ndt::make_type<dynd::ndt::any_kind_type>()}),
      make_comparison_children<func_ptr, CallableType>());
}
}
