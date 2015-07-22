//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/func/callable.hpp>

namespace dynd {
namespace kernels {

  /**
   * Makes a unary reduction ckernel which adds values for the
   * given type id. This is not defined for all type_id values.
   */
  intptr_t make_builtin_sum_reduction_ckernel(void *ckb, intptr_t ckb_offset,
                                              type_id_t tid,
                                              kernel_request_t kernreq);

  /**
   * Makes a unary reduction callable for the requested
   * type id.
   */
  nd::callable make_builtin_sum_reduction_callable(type_id_t tid);

  /**
   * Makes a 1D sum callable.
   * (Fixed * <tid>) -> <tid>
   */
  nd::callable make_builtin_sum1d_callable(type_id_t tid);

  /**
   * Makes a 1D mean callable.
   * (Fixed * <tid>) -> <tid>
   */
  nd::callable make_builtin_mean1d_callable(type_id_t tid, intptr_t minp);

  intptr_t make_strided_reduction_ckernel(void *ckb, intptr_t ckb_offset);

  nd::callable make_strided_reduction_callable();
}
} // namespace dynd::kernels
