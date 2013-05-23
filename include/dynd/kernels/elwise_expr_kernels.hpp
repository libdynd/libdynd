//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_EXPR_KERNELS_HPP_
#define _DYND__ELWISE_EXPR_KERNELS_HPP_

#include <dynd/kernels/expr_kernel_generator.hpp>

namespace dynd {

size_t make_elwise_dimension_expr_kernel(hierarchical_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                size_t src_count, const dtype *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler);

} // namespace dynd

#endif // _DYND__ELWISE_EXPR_KERNELS_HPP_
