//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

namespace dynd {

/**
 * Converts a single UTF8 string to an integer type.
 */
DYND_API void assign_utf8_string_to_builtin(type_id_t dst_type_id, char *dst,
                                   const char *str_begin, const char *str_end,
                                   const eval::eval_context *ectx);

/**
 * Makes a kernel which converts strings to values of a builtin type id.
 */
DYND_API size_t make_builtin_to_string_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_string_dt,
    const char *dst_arrmeta, type_id_t src_type_id, kernel_request_t kernreq,
    const eval::eval_context *ectx);

/**
 * Makes a kernel which converts values of a builtin type id to strings.
 */
DYND_API size_t make_string_to_builtin_assignment_kernel(
    void *ckb, intptr_t ckb_offset, type_id_t dst_type_id,
    const ndt::type &src_string_dt, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx);

} // namespace dynd
