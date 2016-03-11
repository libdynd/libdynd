//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/type.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

struct pod_memory_block : memory_block_data {
  size_t data_size;
  intptr_t data_alignment;
  intptr_t m_total_allocated_capacity;
  /** The malloc'd memory */
  std::vector<char *> m_memory_handles;
  /** The current malloc'd memory being doled out */
  char *m_memory_begin, *m_memory_current, *m_memory_end;

  /**
   * Allocates some new memory from which to dole out
   * more. Adds it to the memory handles vector.
   */
  void append_memory(intptr_t capacity_bytes)
  {
    m_memory_handles.push_back(NULL);
    m_memory_begin = reinterpret_cast<char *>(malloc(capacity_bytes));
    m_memory_handles.back() = m_memory_begin;
    if (m_memory_begin == NULL) {
      m_memory_handles.pop_back();
      throw std::bad_alloc();
    }
    m_memory_current = m_memory_begin;
    m_memory_end = m_memory_current + capacity_bytes;
    m_total_allocated_capacity += capacity_bytes;
  }

  pod_memory_block(size_t data_size, intptr_t data_alignment, intptr_t initial_capacity_bytes)
      : memory_block_data(1, pod_memory_block_type), data_size(data_size), data_alignment(data_alignment),
        m_total_allocated_capacity(0), m_memory_handles()
  {
    append_memory(initial_capacity_bytes);
  }

  ~pod_memory_block()
  {
    for (size_t i = 0, i_end = m_memory_handles.size(); i != i_end; ++i) {
      free(m_memory_handles[i]);
    }
  }
};

/**
 * Creates a memory block which can be used to allocate POD output memory
 * for blockref types.
 *
 * The initial capacity can be set if a good estimate is known.
 */
DYNDT_API intrusive_ptr<memory_block_data> make_pod_memory_block(const ndt::type &tp,
                                                                intptr_t initial_capacity_bytes = 2048);

DYNDT_API void pod_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                            const std::string &indent);

} // namespace dynd
