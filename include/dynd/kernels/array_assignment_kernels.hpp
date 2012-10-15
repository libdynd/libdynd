//
// Copyright (C) 2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ARRAY_ASSIGNMENT_KERNELS_HPP_
#define _DND__ARRAY_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/unary_kernel_instance.hpp>
#include <dynd/dtype_assign.hpp>

namespace dynd {

/**
 * Gets a kernel which assigns blockref arrays
 */
void get_blockref_array_assignment_kernel(const dtype& dst_element_type,
                const dtype& src_element_type,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel);

} // namespace dynd

#endif // _DND__ARRAY_ASSIGNMENT_KERNELS_HPP_
