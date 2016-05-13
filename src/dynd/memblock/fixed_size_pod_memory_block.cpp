//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cstdlib>

#include <dynd/memblock/fixed_size_pod_memory_block.hpp>

using namespace std;
using namespace dynd;

namespace dynd {
namespace detail {

  void free_fixed_size_pod_memory_block(memory_block_data *memblock) { delete[] reinterpret_cast<char *>(memblock); }
}
} // namespace dynd::detail

intrusive_ptr<memory_block_data> dynd::make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment,
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

void dynd::fixed_size_pod_memory_block_debug_print(const memory_block_data *DYND_UNUSED(memblock),
                                                   std::ostream &DYND_UNUSED(o),
                                                   const std::string &DYND_UNUSED(indent)) {}
