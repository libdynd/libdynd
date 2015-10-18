//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/iter/string_iter.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/array.hpp>

using namespace dynd;
using namespace std;

static void transcode_string_iter_destructor(dim_iter *self)
{
  // Free the reference of the element type
  base_type_xdecref(self->eltype);
  // Free the temporary buffer
  void *buf = reinterpret_cast<void *>(const_cast<char *>(self->data_ptr));
  if (buf != NULL) {
    free(buf);
  }
  // Free the reference owning the data
  memory_block_data *memblock = reinterpret_cast<memory_block_data *>(self->custom[7]);
  if (memblock != NULL) {
    memory_block_decref(memblock);
  }
}

static int transcode_string_iter_next(dim_iter *self)
{
  intptr_t i = static_cast<intptr_t>(self->custom[0]);
  intptr_t size = static_cast<intptr_t>(self->custom[1]);
  intptr_t charsize = static_cast<intptr_t>(self->custom[3]);
  if (i < size) {
    const char *begin = reinterpret_cast<const char *>(self->custom[2]);
    const char *end = begin + size;
    begin += i;
    char *buf_begin = const_cast<char *>(self->data_ptr);
    char *buf_end = buf_begin + static_cast<intptr_t>(self->custom[4]);
    next_unicode_codepoint_t next_fn = reinterpret_cast<next_unicode_codepoint_t>(self->custom[5]);
    append_unicode_codepoint_t append_fn = reinterpret_cast<append_unicode_codepoint_t>(self->custom[6]);
    uint32_t cp;

    // Go until there aren't enough characters to hold a large utf8 char
    while (begin < end && buf_begin + 5 <= buf_end) {
      cp = next_fn(begin, end);
      append_fn(cp, buf_begin, buf_end);
    }

    // Update the index we've arrived at
    self->custom[0] = begin - reinterpret_cast<const char *>(self->custom[2]);

    // Update the buffer range in the iter
    // TODO: Make charsize-specific next functions (for size 1, 2, 4)
    self->data_elcount = (buf_begin - const_cast<char *>(self->data_ptr)) / charsize;
    return 1;
  } else {
    self->data_elcount = 0;
    return 0;
  }
}

static void transcode_string_iter_seek_fixed_encoding(dim_iter *self, intptr_t i)
{
  // Set the index to where we want to seek, and use the `next`
  // function to do the rest.

  intptr_t charsize = static_cast<intptr_t>(self->custom[3]);
  self->custom[0] = static_cast<uintptr_t>(i) * charsize;
  transcode_string_iter_next(self);
}

static void transcode_string_iter_seek_var_encoding(dim_iter *self, intptr_t i)
{
  // With a var encoding, only 0 may be provided for i
  if (i == 0) {
    self->custom[0] = 0;
    transcode_string_iter_next(self);
  } else {
    throw runtime_error("dynd string iterator is only restartable, not seekable, provided index must be 0");
  }
}

static dim_iter_vtable transcode_fixed_encoding_string_iter_vt = {
    transcode_string_iter_destructor, transcode_string_iter_next, transcode_string_iter_seek_fixed_encoding};

static dim_iter_vtable transcode_var_encoding_string_iter_vt = {
    transcode_string_iter_destructor, transcode_string_iter_next, transcode_string_iter_seek_var_encoding};

void iter::make_string_iter(dim_iter *out_di, string_encoding_t iter_encoding, string_encoding_t data_encoding,
                            const char *data_begin, const char *data_end, const memory_block_ptr &ref,
                            intptr_t buffer_max_mem, const eval::eval_context *ectx)
{
  ndt::type ctp = char_type_of_encoding(iter_encoding);
  intptr_t datasize = (data_end - data_begin) / string_encoding_char_size_table[data_encoding];
  if (datasize == 0) {
    // Return an empty iterator
    make_empty_dim_iter(out_di, ctp, NULL);
    return;
  } else if (iter_encoding == data_encoding) {
    // With no encoding change, it's a simple strided iteration
    make_strided_dim_iter(out_di, ctp, NULL, data_begin, datasize, ctp.get_data_size(), ref);
    return;
  } else {
    intptr_t charsize = string_encoding_char_size_table[iter_encoding];
    intptr_t bufsize = buffer_max_mem / charsize;
    if (!is_variable_length_string_encoding(iter_encoding) && datasize <= bufsize) {
      // If the whole input string fits in the output max buffer size, make a copy
      nd::array tmp = nd::empty(ndt::string_type::make());
      string_type_arrmeta md;
      md.blockref = ref.get();
      string d;
      d.assign(const_cast<char *>(data_begin), data_end - data_begin);
      tmp.val_assign(ndt::string_type::make(), reinterpret_cast<const char *>(&md), reinterpret_cast<const char *>(&d),
                     ectx);
      tmp.get_type().extended<ndt::string_type>()->make_string_iter(out_di, iter_encoding, tmp.get_arrmeta(),
                                                                    tmp.get_readonly_originptr(),
                                                                    tmp.get_data_memblock(), buffer_max_mem, ectx);
      return;
    }
    // Create an iterator which transcodes
    if (is_variable_length_string_encoding(data_encoding)) {
      out_di->vtable = &transcode_var_encoding_string_iter_vt;
      // variable-sized encodings are not seekable
      out_di->flags = dim_iter_restartable | dim_iter_contiguous;
    } else {
      out_di->vtable = &transcode_fixed_encoding_string_iter_vt;
      // fixed-sized encodings are seekable
      out_di->flags = dim_iter_seekable | dim_iter_restartable | dim_iter_contiguous;
    }
    void *buf = malloc(bufsize * charsize);
    if (buf == NULL) {
      throw bad_alloc();
    }
    out_di->data_ptr = reinterpret_cast<const char *>(buf);
    out_di->data_elcount = 0;
    out_di->data_stride = charsize;
    out_di->eltype = char_type_of_encoding(iter_encoding).release();
    out_di->el_arrmeta = NULL;
    // The custom fields are where we place the data needed for seeking
    // and the reference object.
    out_di->custom[0] = 0;                                                     // The next index to buffer
    out_di->custom[1] = datasize;                                              // Number of characters in buffer
    out_di->custom[2] = reinterpret_cast<uintptr_t>(data_begin);               // The input data;
    out_di->custom[3] = charsize;                                              // size of char in the input data
    out_di->custom[4] = reinterpret_cast<uintptr_t>(buf) + bufsize * charsize; // buffer end
    out_di->custom[5] = reinterpret_cast<uintptr_t>(get_next_unicode_codepoint_function(data_encoding, ectx->errmode));
    out_di->custom[6] =
        reinterpret_cast<uintptr_t>(get_append_unicode_codepoint_function(iter_encoding, ectx->errmode));
    if (ref.get() != NULL) {
      memory_block_incref(ref.get());
      out_di->custom[7] = reinterpret_cast<uintptr_t>(ref.get());
    } else {
      out_di->custom[7] = 0;
    }
  }
}
