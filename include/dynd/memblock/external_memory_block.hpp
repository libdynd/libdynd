//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

typedef void (*external_memory_block_free_t)(void *);

struct external_memory_block : memory_block_data {
  /** A void pointer for the external object */
  void *m_object;
  /** A function which frees the external object */
  external_memory_block_free_t m_free_fn;

  external_memory_block(void *object, external_memory_block_free_t free_fn)
      : memory_block_data(1, external_memory_block_type), m_object(object), m_free_fn(free_fn)
  {
  }
};

/**
 * Creates a memory block which is a reference to an external object.
 */
DYNDT_API intrusive_ptr<memory_block_data> make_external_memory_block(void *object,
                                                                     external_memory_block_free_t free_fn);

DYNDT_API void external_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                                 const std::string &indent);

} // namespace dynd
