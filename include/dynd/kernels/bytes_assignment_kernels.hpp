//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BYTES_ASSIGNMENT_KERNELS_HPP_
#define _DYND__BYTES_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/dtype_assign.hpp>

namespace dynd {

/**
 * Gets a kernel which copies blockref bytes.
 */
void get_blockref_bytes_assignment_kernel(size_t dst_alignment,
                size_t src_alignment,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Gets a kernel which copies fixed-size bytes to bytes.
 */
void get_fixedbytes_to_blockref_bytes_assignment_kernel(size_t dst_alignment,
                intptr_t src_element_size, size_t src_alignment,
                kernel_instance<unary_operation_pair_t>& out_kernel);

} // namespace dynd

#endif // _DYND__BYTES_ASSIGNMENT_KERNELS_HPP_
