//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BYTES_ASSIGNMENT_KERNELS_HPP_
#define _DYND__BYTES_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/dtype_assign.hpp>

namespace dynd {

/**
 * Makes a kernel which copies blockref bytes.
 */
size_t make_blockref_bytes_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                size_t dst_alignment, const char *dst_metadata,
                size_t src_alignment, const char *src_metadata,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which copies fixed-size bytes to bytes.
 */
size_t make_fixedbytes_to_blockref_bytes_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                size_t dst_alignment, const char *dst_metadata,
                intptr_t src_element_size, size_t src_alignment,
                const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__BYTES_ASSIGNMENT_KERNELS_HPP_
