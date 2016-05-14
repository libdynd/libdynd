//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/buffer.hpp>
#include <dynd/memblock/fixed_size_pod_memory_block.hpp>
#include <dynd/memblock/zeroinit_memory_block.hpp>

using namespace std;
using namespace dynd;

intrusive_ptr<nd::memory_block_data> nd::make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment,
                                                                          char **out_datapointer) {
  // Calculate the aligned starting point for the data
  intptr_t start =
      (intptr_t)(((uintptr_t)sizeof(memory_block_data) + (uintptr_t)(alignment - 1)) & ~((uintptr_t)(alignment - 1)));
  // Allocate it
  fixed_size_pod_memory_block *result =
      new (start + size_bytes - sizeof(fixed_size_pod_memory_block)) fixed_size_pod_memory_block();
  // Give back the data pointer
  *out_datapointer = reinterpret_cast<char *>(result) + start;
  // Use placement new to initialize and return the memory block
  return intrusive_ptr<memory_block_data>(result, false);
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
