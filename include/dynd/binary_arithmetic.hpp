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

typedef type_id_sequence<uint8_id, uint16_id, uint32_id, uint64_id, int8_id, int16_id, int32_id, int64_id, float32_id,
                         float64_id, complex_float32_id, complex_float64_id>
    binop_ids;

typedef type_sequence<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double,
                      dynd::complex<float>, dynd::complex<double>>
    binop_types;

template <template <typename, typename> class KernelType, template <typename, typename> class Condition,
          typename TypeSequence>
nd::callable make_binary_arithmetic() {
  const ndt::type &tp = ndt::type("(Any, Any) -> Any");

  auto dispatcher = nd::callable::make_all_if<KernelType, Condition, TypeSequence, TypeSequence>();
  dispatcher.insert({{{option_id, any_kind_id}, nd::functional::forward_na<0>(ndt::type("Any"))},
                     {{any_kind_id, option_id}, nd::functional::forward_na<1>(ndt::type("Any"))},
                     {{option_id, option_id}, nd::functional::forward_na<0, 1>(ndt::type("Any"))},
                     {{dim_kind_id, scalar_kind_id}, nd::get_elwise()},
                     {{scalar_kind_id, dim_kind_id}, nd::get_elwise()},
                     {{dim_kind_id, dim_kind_id}, nd::get_elwise()}});

  return nd::make_callable<nd::arithmetic_dispatch_callable<2>>(tp, dispatcher);
}

} // anonymous namespace
