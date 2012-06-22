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
 * Typedef for getting the next unicode codepoint from a string of a particular
 * encoding.
 *
 * On entry, this function assumes that 'it' and 'end' are appropriately aligned
 * and that (it < end). The variable 'it' is updated in-place to be after the
 * character data representing the returned code point.
 *
 * This function may raise an exception if there is an error.
 */
typedef uint32_t (*next_unicode_codepoint_t)(const char *&it, const char *end);

/**
 * Typedef for appending a unicode codepoint to a string of a particular
 * encoding.
 *
 * On entry, this function assumes that 'it' and 'end' are appropriately aligned
 * and that (it < end). The variable 'it' is updated in-place to be after the
 * character data representing the appended code point.
 *
 * This function may raise an exception if there is an error.
 */
typedef void (*append_unicode_codepoint_t)(uint32_t cp, char *&it, char *end);

next_unicode_codepoint_t get_next_unicode_codepoint_function(string_encoding_t encoding, assign_error_mode errmode);
append_unicode_codepoint_t get_append_unicode_codepoint_function(string_encoding_t encoding, assign_error_mode errmode);

/**
 * Gets a kernel which converts strings of a fixed size from one codec to another.
 */
void get_fixedstring_encoding_kernel(intptr_t dst_element_size, string_encoding_t dst_encoding,
                intptr_t src_element_size, string_encoding_t src_encoding,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel);

} // namespace dnd

#endif // _DND__STRING_ENCODING_KERNELS_HPP_