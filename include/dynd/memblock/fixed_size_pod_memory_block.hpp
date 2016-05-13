//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

class fixed_size_pod_memory_block : public memory_block_data {
public:
  void debug_print(std::ostream &o, const std::string &indent) {
    o << indent << "------ memory_block at " << static_cast<const void *>(this) << "\n";
    o << indent << " reference count: " << m_use_count << "\n";
    o << indent << "------" << std::endl;
  }

  static void *operator new(size_t size, size_t extra_size) { return ::operator new(size + extra_size); }

  static void operator delete(void *ptr) { return ::operator delete(ptr); }

  static void operator delete(void *ptr, size_t DYND_UNUSED(extra_size)) { return ::operator delete(ptr); }
};

/**
 * Creates a memory block of a pre-determined fixed size. A pointer to the
 * memory allocated for data is placed in the output parameter.
 */
DYNDT_API intrusive_ptr<memory_block_data> make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment,
                                                                            char **out_datapointer);

} // namespace dynd
