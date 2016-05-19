//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arithmetic.hpp>
#include <dynd/callables/compound_arithmetic_dispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/arithmetic.hpp>

using namespace std;
using namespace dynd;

namespace {

typedef type_id_sequence<uint8_id, uint16_id, uint32_id, uint64_id, int8_id, int16_id, int32_id, int64_id, float32_id,
                         float64_id, complex_float32_id, complex_float64_id>
    binop_ids;
typedef type_sequence<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double,
                      dynd::complex<float>, dynd::complex<double>>
    binop_types;

template <template <typename, typename> class KernelType, typename TypeSequence>
nd::callable make_compound_arithmetic() {
  const ndt::type &tp = ndt::type("(Any, Any) -> Any");
  auto dispatcher = nd::callable::make_all<KernelType, TypeSequence, TypeSequence>();

  for (type_id_t i0 : i2a<binop_ids>()) {
    for (type_id_t i1 : i2a<dim_ids>()) {
      dispatcher.insert({{i0, i1}, nd::get_elwise()});
    }
  }

  for (type_id_t i0 : i2a<dim_ids>()) {
    typedef typename join<binop_ids, dim_ids>::type broadcast_ids;
    for (type_id_t i1 : i2a<broadcast_ids>()) {
      dispatcher.insert({{i0, i1}, nd::get_elwise()});
    }
  }

  return nd::make_callable<nd::compound_arithmetic_dispatch_callable>(tp, dispatcher);
}

} // anonymous namespace
