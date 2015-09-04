//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/typed_data_assign.hpp>

namespace dynd {

/**
 * Makes a kernel which copies blockref bytes.
 */
DYND_API size_t make_blockref_bytes_assignment_kernel(
                void *ckb, intptr_t ckb_offset,
                size_t dst_alignment, const char *dst_arrmeta,
                size_t src_alignment, const char *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx);

/**
 * Makes a kernel which copies fixed-size bytes to bytes.
 */
DYND_API size_t make_fixed_bytes_to_blockref_bytes_assignment_kernel(
                void *ckb, intptr_t ckb_offset,
                size_t dst_alignment, const char *dst_arrmeta,
                intptr_t src_element_size, size_t src_alignment,
                kernel_request_t kernreq, const eval::eval_context *ectx);

} // namespace dynd
