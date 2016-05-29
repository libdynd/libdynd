//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arithmetic.hpp>
#include <dynd/callables/arithmetic_dispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/arithmetic.hpp>
#include <dynd/types/scalar_kind_type.hpp>

using namespace std;
using namespace dynd;

namespace {

static std::vector<ndt::type> func_ptr(const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc),
                                       const ndt::type *src_tp) {
  return {src_tp[0]};
}

typedef type_sequence<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double,
                      dynd::complex<float>, dynd::complex<double>>
    binop_types;

template <template <typename> class CallableType, template <typename> class Condition, typename TypeSequence>
nd::callable make_unary_arithmetic() {
  dispatcher<func_ptr, 1, nd::callable> dispatcher =
      nd::callable::make_all_if<func_ptr, CallableType, Condition, TypeSequence>();

  const ndt::type &tp = ndt::type("(Any) -> Any");
  for (ndt::type i0 : {ndt::make_type<ndt::fixed_dim_kind_type>(), ndt::make_type<ndt::var_dim_type>()}) {
    dispatcher.insert({{i0}, nd::get_elwise()});
  }

  return nd::make_callable<nd::arithmetic_dispatch_callable<func_ptr, 1>>(tp, dispatcher);
}

} // anonymous namespace
