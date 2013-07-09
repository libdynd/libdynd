//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRING_COMPARISON_KERNELS_HPP_
#define _DYND__STRING_COMPARISON_KERNELS_HPP_

#include <dynd/kernels/comparison_kernels.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

/**
 * Makes a kernel which compares fixedstrings.
 *
 * \param string_size  The number of characters (1, 2, or 4-bytes each) in the string.
 * \param encoding  The encoding of the string.
 */
size_t make_fixedstring_comparison_kernel(
                hierarchical_kernel *out, size_t offset_out,
                size_t string_size, string_encoding_t encoding,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which compares blockref strings.
 *
 * \param encoding  The encoding of the string.
 */
size_t make_string_comparison_kernel(
                hierarchical_kernel *out, size_t offset_out,
                string_encoding_t encoding,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which compares two .
 *
 * \param encoding  The encoding of the string.
 */
size_t make_general_string_comparison_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& src0_dt, const char *src0_metadata,
                const ndt::type& src1_dt, const char *src1_metadata,
                comparison_type_t comptype,
                const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__STRING_COMPARISON_KERNELS_HPP_
