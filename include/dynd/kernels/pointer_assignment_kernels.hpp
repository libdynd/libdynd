//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {

/**
 * Makes a kernel which assigns the pointer to a built-in value.
 */
DYND_API size_t make_builtin_value_to_pointer_assignment_kernel(
    void *ckb, intptr_t ckb_offset, type_id_t tp_id,
    kernel_request_t kernreq);

/**
 * Makes a kernel which assigns the pointer to a value.
 */
DYND_API size_t make_value_to_pointer_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &tp,
    kernel_request_t kernreq);

} // namespace dynd
