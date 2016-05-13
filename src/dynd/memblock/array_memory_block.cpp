//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/exceptions.hpp>
#include <dynd/memblock/array_memory_block.hpp>
#include <dynd/types/base_memory_type.hpp>

using namespace std;
using namespace dynd;

intrusive_ptr<memory_block_data> dynd::make_array_memory_block(const ndt::type &tp, size_t arrmeta_size,
                                                               size_t extra_size, size_t extra_alignment,
                                                               char **out_extra_ptr) {
  size_t extra_offset = inc_to_alignment(sizeof(array_preamble) + arrmeta_size, extra_alignment);
  array_preamble *result = new (extra_offset + extra_size - sizeof(array_preamble)) array_preamble(tp, arrmeta_size);
  // Return a pointer to the extra allocated memory
  *out_extra_ptr = reinterpret_cast<char *>(result) + extra_offset;
  return intrusive_ptr<memory_block_data>(result, false);
}

intrusive_ptr<memory_block_data> dynd::shallow_copy_array_memory_block(const intrusive_ptr<memory_block_data> &ndo) {
  // Allocate the new memory block.
  const array_preamble *preamble = reinterpret_cast<const array_preamble *>(ndo.get());
  size_t arrmeta_size = 0;
  if (!preamble->tp.is_builtin()) {
    arrmeta_size = preamble->tp->get_arrmeta_size();
  }
  intrusive_ptr<memory_block_data> result = make_array_memory_block(preamble->tp, arrmeta_size);
  array_preamble *result_preamble = reinterpret_cast<array_preamble *>(result.get());

  // Clone the data pointer
  result_preamble->data = preamble->data;
  result_preamble->owner = preamble->owner;
  if (!result_preamble->owner) {
    result_preamble->owner = ndo.get();
  }

  // Copy the flags
  result_preamble->flags = preamble->flags;

  if (!preamble->tp.is_builtin()) {
    preamble->tp.extended()->arrmeta_copy_construct(reinterpret_cast<char *>(result.get()) + sizeof(array_preamble),
                                                    reinterpret_cast<const char *>(ndo.get()) + sizeof(array_preamble),
                                                    ndo);
  }

  return result;
}
