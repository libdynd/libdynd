//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRING_ITER_HPP_
#define _DYND__STRING_ITER_HPP_

#include <dynd/dim_iter.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd { namespace iter {

/**
 * Makes a dim_iter for a string. If the iteration
 * encoding is the same as the string encoding, this is just
 * a strided iterator, otherwise it makes an iterator which
 * buffers the encoding transformation.
 *
 * \param out_di  An uninitialized dim_iter object. The function
 *                populates it assuming it is filled with garbage.
 * \param iter_encoding  The string encoding the user of the iterator
 *                       requires.
 * \param data_encoding  The string encoding of the string.
 * \param data_begin  The beginning of the data buffer.
 * \param data_end  One past the end of the data buffer.
 * \param ref  A blockref which owns the string's data.
 */
void make_string_iter(
    dim_iter *out_di, string_encoding_t iter_encoding,
    string_encoding_t data_encoding,
    const char *data_begin, const char *data_end,
    const memory_block_ptr& ref,
    intptr_t buffer_max_mem,
    const eval::eval_context *ectx = &eval::default_eval_context);

}} // namespace dynd::iter

#endif // _DYND__STRING_ITER_HPP_

