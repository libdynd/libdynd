//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EXPRESSION_ASSIGNMENT_KERNELS_HPP_
#define _DYND__EXPRESSION_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/assignment_kernels.hpp>

namespace dynd {

/** The number of elements buffered when chaining expressions */
#define DYND_BUFFER_CHUNK_SIZE ((size_t)128)

/**
 * Makes a kernel which does an assignment when
 * at least one of dst_dt and src_dt is an
 * expression_kind dtype.
 */
size_t make_expression_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__EXPRESSION_ASSIGNMENT_KERNELS_HPP_

