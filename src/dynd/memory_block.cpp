//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/memblock/objectarray_memory_block.hpp>
#include <dynd/memblock/zeroinit_memory_block.hpp>
#include <dynd/buffer.hpp>

using namespace std;
using namespace dynd;

nd::memory_block nd::make_objectarray_memory_block(const ndt::type &dt, const char *arrmeta, intptr_t stride,
                                                   intptr_t initial_count, size_t arrmeta_size) {
  objectarray_memory_block *pmb = new objectarray_memory_block(dt, arrmeta_size, arrmeta, stride, initial_count);
  return memory_block(reinterpret_cast<memory_block_data *>(pmb), false);
}

nd::memory_block nd::make_zeroinit_memory_block(const ndt::type &element_tp, intptr_t initial_capacity_bytes) {
  // This is a temporary hack until the new bytes and string types are working
  size_t data_size;
  switch (element_tp.get_id()) {
  case bytes_id:
  case string_id:
    data_size = 1;
    break;
  default:
    data_size = element_tp.get_default_data_size();
  }

  zeroinit_memory_block *pmb =
      new zeroinit_memory_block(data_size, element_tp.get_data_alignment(), initial_capacity_bytes);
  return memory_block(reinterpret_cast<memory_block_data *>(pmb), false);
}
