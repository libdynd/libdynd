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
  dispatcher<1, nd::callable> dispatcher = nd::callable::make_all_if<CallableType, Condition, TypeSequence>(func_ptr);

  const ndt::type &tp = ndt::type("(Any) -> Any");
  dispatcher.insert({{ndt::type("Fixed * Any")}, nd::get_elwise(ndt::type("(Fixed * Any) -> Any"))});
  dispatcher.insert({{ndt::type("var * Any")}, nd::get_elwise(ndt::type("(var * Any) -> Any"))});

  return nd::make_callable<nd::arithmetic_dispatch_callable<func_ptr, 1>>(tp, dispatcher);
}

} // anonymous namespace
