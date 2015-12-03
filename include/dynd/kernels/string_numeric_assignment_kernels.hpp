//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

namespace dynd {

/**
 * Makes a kernel which converts strings to values of a builtin type id.
 */
DYND_API size_t make_builtin_to_string_assignment_kernel(void *ckb, intptr_t ckb_offset, const ndt::type &dst_string_dt,
                                                         const char *dst_arrmeta, type_id_t src_type_id,
                                                         kernel_request_t kernreq, const eval::eval_context *ectx);

} // namespace dynd
