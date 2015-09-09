//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/string_encodings.hpp>

namespace dynd {

/**
 * Makes a kernel which converts strings of a fixed size from one codec to another.
 */
DYND_API size_t make_fixed_string_assignment_kernel(
    void *ckb, intptr_t ckb_offset, intptr_t dst_data_size,
    string_encoding_t dst_encoding, intptr_t src_data_size,
    string_encoding_t src_encoding, kernel_request_t kernreq,
    const eval::eval_context *ectx);

/**
 * Makes a kernel which converts a single char from one encoding to another.
 */
DYND_API size_t make_char_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                            string_encoding_t dst_encoding,
                                            string_encoding_t src_encoding,
                                            kernel_request_t kernreq,
                                            const eval::eval_context *ectx);

/**
 * Makes a kernel which converts blockref strings from one codec to another.
 */
DYND_API size_t make_blockref_string_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
    string_encoding_t dst_encoding, const char *src_arrmeta,
    string_encoding_t src_encoding, kernel_request_t kernreq,
    const eval::eval_context *ectx);

/**
 * Makes a kernel which converts strings of a fixed size into blockref strings.
 */
DYND_API size_t make_fixed_string_to_blockref_string_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
    string_encoding_t dst_encoding, intptr_t src_element_size,
    string_encoding_t src_encoding, kernel_request_t kernreq,
    const eval::eval_context *ectx);

/**
 * Makes a kernel which converts blockref strings into strings of a fixed size.
 */
DYND_API size_t make_blockref_string_to_fixed_string_assignment_kernel(
    void *ckb, intptr_t ckb_offset, intptr_t dst_data_size,
    string_encoding_t dst_encoding, string_encoding_t src_encoding,
    kernel_request_t kernreq, const eval::eval_context *ectx);

/**
 * Makes a kernel which converts values to string using the
 * stream output operator defined on the type objects.
 */
DYND_API size_t make_any_to_string_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx);

} // namespace dynd
