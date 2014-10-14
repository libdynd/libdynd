//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__POINTER_ASSIGNMENT_KERNELS_HPP_
#define _DYND__POINTER_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernels.hpp>

namespace dynd {

/**
 * Makes a kernel which assigns the pointer to a built-in value.
 */
size_t make_builtin_value_to_pointer_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, type_id_t tp_id,
    kernel_request_t kernreq);

/**
 * Makes a kernel which assigns the pointer to a value.
 */
size_t make_value_to_pointer_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &tp,
    kernel_request_t kernreq);

} // namespace dynd

#endif // _DYND__POINTER_ASSIGNMENT_KERNELS_HPP_

