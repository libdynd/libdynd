//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arithmetic.hpp>
#include <dynd/callables/multidispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/arithmetic.hpp>

using namespace std;
using namespace dynd;

namespace {

typedef type_sequence<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double,
                      dynd::complex<float>, dynd::complex<double>>
    binop_types;

template <template <typename, typename> class KernelType, typename TypeSequence>
nd::callable make_compound_arithmetic() {
  const ndt::type &tp = ndt::make_type<ndt::callable_type>(
      ndt::make_type<ndt::any_kind_type>(),
      ndt::make_type<ndt::tuple_type>({ndt::make_type<ndt::any_kind_type>(), ndt::make_type<ndt::any_kind_type>()}),
      ndt::make_type<ndt::struct_type>());

  auto dispatcher = nd::callable::make_all<KernelType, TypeSequence, TypeSequence>(
      [](const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp) -> std::vector<ndt::type> {
        return {dst_tp, src_tp[0]};
      });

  static const std::vector<ndt::type> binop_ids = {ndt::make_type<uint8_t>(),
                                                   ndt::make_type<uint16_t>(),
                                                   ndt::make_type<uint32_t>(),
                                                   ndt::make_type<uint64_t>(),
                                                   ndt::make_type<int8_t>(),
                                                   ndt::make_type<int16_t>(),
                                                   ndt::make_type<int32_t>(),
                                                   ndt::make_type<int64_t>(),
                                                   ndt::make_type<float>(),
                                                   ndt::make_type<double>(),
                                                   ndt::make_type<dynd::complex<float>>(),
                                                   ndt::make_type<dynd::complex<double>>()};

  dispatcher.insert(nd::get_elwise(ndt::make_type<ndt::callable_type>(
      ndt::make_type<ndt::any_kind_type>(),
      ndt::make_type<ndt::tuple_type>({ndt::make_type<ndt::scalar_kind_type>(),
                                       ndt::make_type<ndt::dim_kind_type>(ndt::make_type<ndt::any_kind_type>())}),
      ndt::make_type<ndt::struct_type>())));
  dispatcher.insert(nd::get_elwise(ndt::make_type<ndt::callable_type>(
      ndt::make_type<ndt::any_kind_type>(),
      ndt::make_type<ndt::tuple_type>({ndt::make_type<ndt::dim_kind_type>(ndt::make_type<ndt::any_kind_type>()),
                                       ndt::make_type<ndt::scalar_kind_type>()}),
      ndt::make_type<ndt::struct_type>())));
  dispatcher.insert(nd::get_elwise(ndt::make_type<ndt::callable_type>(
      ndt::make_type<ndt::any_kind_type>(),
      ndt::make_type<ndt::tuple_type>({ndt::make_type<ndt::dim_kind_type>(ndt::make_type<ndt::any_kind_type>()),
                                       ndt::make_type<ndt::dim_kind_type>(ndt::make_type<ndt::any_kind_type>())}),
      ndt::make_type<ndt::struct_type>())));

  return nd::make_callable<nd::multidispatch_callable<2>>(tp, dispatcher);
}

} // anonymous namespace
