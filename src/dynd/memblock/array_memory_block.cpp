//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/memblock/array_memory_block.hpp>
#include <dynd/types/base_memory_type.hpp>

using namespace std;
using namespace dynd;

nd::array dynd::make_array_memory_block(const ndt::type &tp, size_t arrmeta_size, size_t extra_size,
                                        size_t extra_alignment, char **out_extra_ptr) {
  size_t extra_offset = inc_to_alignment(sizeof(array_preamble) + arrmeta_size, extra_alignment);
  array_preamble *result = new (extra_offset + extra_size - sizeof(array_preamble)) array_preamble(tp, arrmeta_size);
  // Return a pointer to the extra allocated memory
  *out_extra_ptr = reinterpret_cast<char *>(result) + extra_offset;
  return nd::array(result, false);
}

nd::array dynd::shallow_copy_array_memory_block(const nd::array &ndo) {
  // Allocate the new memory block.
  size_t arrmeta_size = 0;
  if (!ndo->tp.is_builtin()) {
    arrmeta_size = ndo->tp->get_arrmeta_size();
  }

  nd::array result = make_array_memory_block(ndo->tp, arrmeta_size);
  // Clone the data pointer
  result->data = ndo->data;
  result->owner = ndo->owner;
  if (!result->owner) {
    result->owner = ndo.get();
  }

  // Copy the flags
  result->flags = ndo->flags;

  if (!ndo->tp.is_builtin()) {
    ndo->tp.extended()->arrmeta_copy_construct(result->metadata(), ndo->metadata(),
                                               intrusive_ptr<memory_block_data>(ndo.get(), true));
  }

  return result;
}
