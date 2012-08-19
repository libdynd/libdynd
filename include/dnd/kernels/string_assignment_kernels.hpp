//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__STRING_ENCODING_KERNELS_HPP_
#define _DND__STRING_ENCODING_KERNELS_HPP_

#include <dnd/kernels/unary_kernel_instance.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/string_encodings.hpp>
#include <stdint.h>

namespace dnd {

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

} // namespace dnd

#endif // _DND__STRING_ENCODING_KERNELS_HPP_
