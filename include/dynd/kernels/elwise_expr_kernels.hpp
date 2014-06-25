//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_EXPR_KERNELS_HPP_
#define _DYND__ELWISE_EXPR_KERNELS_HPP_

#include <dynd/kernels/expr_kernel_generator.hpp>

namespace dynd {

size_t make_elwise_dimension_expr_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                size_t src_count, const ndt::type *src_dt, const char *const*src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler);

} // namespace dynd

#endif // _DYND__ELWISE_EXPR_KERNELS_HPP_
