//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRING_COMPARISON_KERNELS_HPP_
#define _DYND__STRING_COMPARISON_KERNELS_HPP_

#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

/**
 * Gets a kernel which compares fixedstrings.
 *
 * \param string_size  The number of characters (1, 2, or 4-bytes each) in the string.
 * \param encoding  The encoding of the string.
 * \param out_kernel  The kernel produced.
 */
void get_fixedstring_comparison_kernel(intptr_t string_size, string_encoding_t encoding,
                kernel_instance<compare_operations_t>& out_kernel);

/**
 * Gets a kernel which compares blockref strings.
 *
 * \param encoding  The encoding of the string.
 * \param out_kernel  The kernel produced.
 */
void get_string_comparison_kernel(string_encoding_t encoding,
                kernel_instance<compare_operations_t>& out_kernel);

} // namespace dynd

#endif // _DYND__STRING_COMPARISON_KERNELS_HPP_
