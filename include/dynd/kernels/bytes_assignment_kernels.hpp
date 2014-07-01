//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BYTES_ASSIGNMENT_KERNELS_HPP_
#define _DYND__BYTES_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

/**
 * Makes a kernel which copies blockref bytes.
 */
size_t make_blockref_bytes_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                size_t dst_alignment, const char *dst_arrmeta,
                size_t src_alignment, const char *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx);

/**
 * Makes a kernel which copies fixed-size bytes to bytes.
 */
size_t make_fixedbytes_to_blockref_bytes_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                size_t dst_alignment, const char *dst_arrmeta,
                intptr_t src_element_size, size_t src_alignment,
                kernel_request_t kernreq, const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__BYTES_ASSIGNMENT_KERNELS_HPP_
