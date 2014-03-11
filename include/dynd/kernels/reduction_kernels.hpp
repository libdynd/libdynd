//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__REDUCTION_KERNELS_HPP_
#define _DYND__REDUCTION_KERNELS_HPP_

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

namespace dynd { namespace kernels {

/**
 * Makes a unary reduction ckernel which adds values for the
 * given type id. This is not defined for all type_id values.
 */
intptr_t make_builtin_sum_reduction_ckernel(
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                type_id_t tid,
                kernel_request_t kerntype);

}} // namespace dynd::kernels

#endif // _DYND__REDUCTION_KERNELS_HPP_