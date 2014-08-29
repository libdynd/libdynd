//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRING_COMPARISON_KERNELS_HPP_
#define _DYND__STRING_COMPARISON_KERNELS_HPP_

#include <dynd/kernels/comparison_kernels.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

/**
 * Makes a kernel which compares fixedstrings.
 *
 * \param string_size  The number of characters (1, 2, or 4-bytes each) in the string.
 * \param encoding  The encoding of the string.
 */
size_t make_fixedstring_comparison_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                size_t string_size, string_encoding_t encoding,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which compares blockref strings.
 *
 * \param encoding  The encoding of the string.
 */
size_t make_string_comparison_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                string_encoding_t encoding,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which compares two strings of any type.
 *
 */
size_t make_general_string_comparison_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& src0_dt, const char *src0_arrmeta,
                const ndt::type& src1_dt, const char *src1_arrmeta,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__STRING_COMPARISON_KERNELS_HPP_
