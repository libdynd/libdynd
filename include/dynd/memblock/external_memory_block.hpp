//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/memblock/memory_block.hpp>

namespace dynd {
namespace nd {

  typedef void (*external_memory_block_free_t)(void *);

  /**
   * A memory block which is a reference to an external object.
   */
  class external_memory_block : public memory_block_data {
  public:
    /** A void pointer for the external object */
    void *m_object;
    /** A function which frees the external object */
    external_memory_block_free_t m_free_fn;

    external_memory_block(void *object, external_memory_block_free_t free_fn) : m_object(object), m_free_fn(free_fn) {}

    ~external_memory_block() { m_free_fn(m_object); }

    void debug_print(std::ostream &o, const std::string &indent) {
      o << indent << "------ memory_block at " << static_cast<const void *>(this) << "\n";
      o << indent << " reference count: " << m_use_count << "\n";
      o << indent << " object void pointer: " << m_object << "\n";
      o << indent << " free function: " << (const void *)m_free_fn << "\n";
      o << indent << "------" << std::endl;
    }
  };

} // namespace dynd::nd
} // namespace dynd
