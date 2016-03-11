//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/memblock/memory_block.hpp>
#include <dynd/type.hpp>

namespace dynd {

struct memory_chunk {
  char *memory;
  size_t used_count, capacity_count;
};

struct objectarray_memory_block : memory_block_data {
  ndt::type m_dt;
  size_t arrmeta_size;
  const char *m_arrmeta;
  intptr_t m_stride;
  size_t m_total_allocated_count;
  bool m_finalized;
  /** The malloc'd memory */
  std::vector<memory_chunk> m_memory_handles;

  objectarray_memory_block(const ndt::type &dt, size_t arrmeta_size, const char *arrmeta, intptr_t stride,
                           intptr_t initial_count)
      : memory_block_data(1, objectarray_memory_block_type), m_dt(dt), arrmeta_size(arrmeta_size), m_arrmeta(arrmeta),
        m_stride(stride), m_total_allocated_count(0), m_finalized(false), m_memory_handles()
  {
    if ((dt.get_flags() & type_flag_destructor) == 0) {
      std::stringstream ss;
      ss << "Cannot create objectarray memory block with dynd type " << dt;
      ss << " because it does not have a destructor, use a POD memory block instead";
      throw std::runtime_error(ss.str());
    }
    append_memory(initial_count);
  }

  ~objectarray_memory_block()
  {
    for (size_t i = 0, i_end = m_memory_handles.size(); i != i_end; ++i) {
      memory_chunk &mc = m_memory_handles[i];
      m_dt.extended()->data_destruct_strided(m_arrmeta + arrmeta_size, mc.memory, m_stride, mc.used_count);
      free(mc.memory);
    }
  }

  /**
   * Allocates some new memory from which to dole out
   * more. Adds it to the memory handles vector.
   */
  void append_memory(intptr_t count)
  {
    m_memory_handles.push_back(memory_chunk());
    memory_chunk &mc = m_memory_handles.back();
    mc.used_count = 0;
    mc.capacity_count = count;
    char *memory = reinterpret_cast<char *>(malloc(m_stride * count));
    mc.memory = memory;
    if (memory == NULL) {
      m_memory_handles.pop_back();
      throw std::bad_alloc();
    }
    m_total_allocated_count += count;
  }
};

/**
 * Creates a memory block which can be used to allocate zero-initialized
 * object type output memory for blockref types.
 *
 * The initial count of elements can be set if a good estimate is known.
 *
 * \param dt  The data type of the objects to allocate.
 * \param arrmeta  The arrmeta corresponding to the data type for the objects to allocate.
 * \param stride  For objects without a fixed size, the size of the memory to allocate
 *                for each element. This would be typically set to the value for
 *                get_default_data_size() corresponding to default-constructed arrmeta.
 * \param initial_count  The number of elements to allocate at the start.
 */
DYNDT_API intrusive_ptr<memory_block_data> make_objectarray_memory_block(const ndt::type &dt, const char *arrmeta,
                                                                        intptr_t stride, intptr_t initial_count = 64,
                                                                        size_t arrmeta_size = 0);

DYNDT_API void objectarray_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                                    const std::string &indent);

} // namespace dynd
