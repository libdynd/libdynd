//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernels.hpp>

namespace dynd {

/**
 * Makes a kernel which assigns the pointer to a built-in value.
 */
size_t make_builtin_value_to_pointer_assignment_kernel(
    void *ckb, intptr_t ckb_offset, type_id_t tp_id,
    kernel_request_t kernreq);

/**
 * Makes a kernel which assigns the pointer to a value.
 */
size_t make_value_to_pointer_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &tp,
    kernel_request_t kernreq);

} // namespace dynd
