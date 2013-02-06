//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRING_ASSIGNMENT_KERNELS_HPP_
#define _DYND__STRING_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

/**
 * Gets a kernel which converts strings of a fixed size from one codec to another.
 */
void get_fixedstring_assignment_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Gets a kernel which converts blockref strings from one codec to another.
 */
void get_blockref_string_assignment_kernel(string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Gets a kernel which converts strings of a fixed size into blockref strings.
 */
void get_fixedstring_to_blockref_string_assignment_kernel(string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Gets a kernel which converts blockref strings into strings of a fixed size
 .
 */
void get_blockref_string_to_fixedstring_assignment_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Makes a kernel which converts strings of a fixed size from one codec to another.
 */
size_t make_fixedstring_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                intptr_t dst_data_size, string_encoding_t dst_encoding,
                intptr_t src_data_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which converts blockref strings from one codec to another.
 */
size_t make_blockref_string_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const char *dst_metadata, string_encoding_t dst_encoding,
                const char *src_metadata, string_encoding_t src_encoding,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which converts strings of a fixed size into blockref strings.
 */
size_t make_fixedstring_to_blockref_string_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const char *dst_metadata, string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which converts blockref strings into strings of a fixed size.
 */
size_t make_blockref_string_to_fixedstring_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                intptr_t dst_data_size, string_encoding_t dst_encoding,
                string_encoding_t src_encoding,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__STRING_ASSIGNMENT_KERNELS_HPP_
