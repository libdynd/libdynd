//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/date_type.hpp>

namespace dynd {

/**
 * Makes a kernel which converts strings to dates.
 */
size_t make_string_to_date_assignment_kernel(void *ckb,
                                             intptr_t ckb_offset,
                                             const ndt::type &src_string_dt,
                                             const char *src_arrmeta,
                                             kernel_request_t kernreq,
                                             const eval::eval_context *ectx);

/**
 * Makes a kernel which converts dates to strings.
 */
size_t make_date_to_string_assignment_kernel(void *ckb,
                                             intptr_t ckb_offset,
                                             const ndt::type &dst_string_dt,
                                             const char *dst_arrmeta,
                                             kernel_request_t kernreq,
                                             const eval::eval_context *ectx);

} // namespace dynd
