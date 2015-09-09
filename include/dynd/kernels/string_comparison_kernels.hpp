//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/comparison_kernels.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

/**
 * Makes a kernel which compares fixed_strings.
 *
 * \param string_size  The number of characters (1, 2, or 4-bytes each) in the string.
 * \param encoding  The encoding of the string.
 */
DYND_API size_t make_fixed_string_comparison_kernel(
                void *ckb, intptr_t ckb_offset,
                size_t string_size, string_encoding_t encoding,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which compares blockref strings.
 *
 * \param encoding  The encoding of the string.
 */
DYND_API size_t make_string_comparison_kernel(
                void *ckb, intptr_t ckb_offset,
                string_encoding_t encoding,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which compares two strings of any type.
 *
 */
DYND_API size_t make_general_string_comparison_kernel(
                void *ckb, intptr_t ckb_offset,
                const ndt::type& src0_dt, const char *src0_arrmeta,
                const ndt::type& src1_dt, const char *src1_arrmeta,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

} // namespace dynd
