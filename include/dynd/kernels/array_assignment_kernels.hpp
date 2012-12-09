//
// Copyright (C) 2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRAY_ASSIGNMENT_KERNELS_HPP_
#define _DYND__ARRAY_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/dtype_assign.hpp>

namespace dynd {

/**
 * Gets a kernel which assigns blockref arrays
 */
void get_blockref_array_assignment_kernel(const dtype& dst_element_type,
                const dtype& src_element_type,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

} // namespace dynd

#endif // _DYND__ARRAY_ASSIGNMENT_KERNELS_HPP_
