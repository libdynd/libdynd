//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRING_ASSIGNMENT_KERNELS_HPP_
#define _DYND__STRING_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/unary_kernel_instance.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

/**
 * Gets a kernel which converts strings of a fixed size from one codec to another.
 */
void get_fixedstring_assignment_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel);

/**
 * Gets a kernel which converts blockref strings from one codec to another.
 */
void get_blockref_string_assignment_kernel(string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel);

/**
 * Gets a kernel which converts strings of a fixed size into blockref strings.
 */
void get_fixedstring_to_blockref_string_assignment_kernel(string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel);

/**
 * Gets a kernel which converts blockref strings into strings of a fixed size
 .
 */
void get_blockref_string_to_fixedstring_assignment_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel);

} // namespace dynd

#endif // _DYND__STRING_ASSIGNMENT_KERNELS_HPP_
