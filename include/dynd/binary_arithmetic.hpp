//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arithmetic.hpp>
#include <dynd/callables/multidispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/arithmetic.hpp>
#include <dynd/types/scalar_kind_type.hpp>

using namespace std;
using namespace dynd;

namespace {

typedef type_sequence<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double,
                      dynd::complex<float>, dynd::complex<double>>
    binop_types;

inline std::vector<ndt::type> func_ptr(const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc),
                                       const ndt::type *src_tp) {
  return {src_tp[0], src_tp[1]};
}

template <template <typename, typename> class KernelType, template <typename, typename> class Condition,
          typename TypeSequence>
nd::callable make_binary_arithmetic() {
  const ndt::type &tp = ndt::type("(Any, Any) -> Any");

  auto dispatcher = nd::callable::template make_all_if<KernelType, Condition, TypeSequence, TypeSequence>(func_ptr);
  dispatcher.insert({nd::functional::forward_na<0>(ndt::type("Any"), {ndt::type("?Any"), ndt::type("Any")}),
                     nd::functional::forward_na<1>(ndt::type("Any"), {ndt::type("Any"), ndt::type("?Any")}),
                     nd::functional::forward_na<0, 1>(ndt::type("Any"), {ndt::type("?Any"), ndt::type("?Any")}),
                     nd::get_elwise(ndt::type("(Dim, Scalar) -> Any")),
                     nd::get_elwise(ndt::type("(Scalar, Dim) -> Any")), nd::get_elwise(ndt::type("(Dim, Dim) -> Any")),
                     nd::get_elwise(ndt::type("(Scalar, Scalar) -> Any"))});

  return nd::make_callable<nd::multidispatch_callable<2>>(tp, dispatcher);
}

} // anonymous namespace
