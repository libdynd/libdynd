//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/reduction.hpp>
#include <dynd/func/sum.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/sum_kernel.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/callables/sum_dispatch_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::sum::make()
{
  typedef type_id_sequence<int8_id, int16_id, int32_id, int64_id, uint8_id, uint16_id, uint32_id, uint64_id, float16_id,
                           float32_id, float64_id, complex_float32_id, complex_float64_id> arithmetic_ids;

  auto dispatcher = callable::new_make_all<sum_kernel, arithmetic_ids>();
  return functional::reduction(make_callable<sum_dispatch_callable>(
      ndt::callable_type::make(ndt::scalar_kind_type::make(), ndt::scalar_kind_type::make()), dispatcher));
}

DYND_DEFAULT_DECLFUNC_GET(nd::sum)

DYND_API struct nd::sum nd::sum;
