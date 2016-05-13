//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

typedef void (*external_memory_block_free_t)(void *);

class external_memory_block : public memory_block_data {
public:
  /** A void pointer for the external object */
  void *m_object;
  /** A function which frees the external object */
  external_memory_block_free_t m_free_fn;

  external_memory_block(void *object, external_memory_block_free_t free_fn)
      : memory_block_data(1, external_memory_block_type), m_object(object), m_free_fn(free_fn) {}

  ~external_memory_block() { m_free_fn(m_object); }

  void debug_print(std::ostream &o, const std::string &indent);
};

/**
 * Creates a memory block which is a reference to an external object.
 */
DYNDT_API intrusive_ptr<memory_block_data> make_external_memory_block(void *object,
                                                                      external_memory_block_free_t free_fn);

} // namespace dynd
