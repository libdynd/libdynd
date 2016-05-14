//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/memblock/memory_block.hpp>
#include <dynd/type.hpp>

namespace dynd {
namespace nd {

  struct memory_chunk {
    char *memory;
    size_t used_count, capacity_count;
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
        : m_dt(dt), arrmeta_size(arrmeta_size), m_arrmeta(arrmeta), m_stride(stride), m_total_allocated_count(0),
          m_finalized(false), m_memory_handles() {
      if ((dt.get_flags() & type_flag_destructor) == 0) {
        std::stringstream ss;
        ss << "Cannot create objectarray memory block with dynd type " << dt;
        ss << " because it does not have a destructor, use a POD memory block instead";
        throw std::runtime_error(ss.str());
      }
      append_memory(initial_count);
    }

    ~objectarray_memory_block() {
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
    void append_memory(intptr_t count) {
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

    char *alloc(size_t count) {
      //    cout << "allocating " << size_bytes << " of memory with alignment " << alignment << endl;
      // Allocate new POD memory of the requested size and alignment
      memory_chunk *mc = &m_memory_handles.back();
      if (mc->capacity_count - mc->used_count < count) {
        append_memory(std::max(m_total_allocated_count, count));
        mc = &m_memory_handles.back();
      }

      char *result = mc->memory + m_stride * mc->used_count;
      mc->used_count += count;
      if ((m_dt.get_flags() & type_flag_zeroinit) != 0) {
        memset(result, 0, m_stride * count);
      } else {
        // TODO: Add a default data constructor to base_type
        //       as well, with a flag for it
        std::stringstream ss;
        ss << "Expected objectarray data to be zeroinit, but is not with dynd type " << m_dt;
        throw std::runtime_error(ss.str());
      }
      return result;
    }

    char *resize(char *previous_allocated, size_t count) {
      memory_chunk *mc = &m_memory_handles.back();
      size_t previous_index = (previous_allocated - mc->memory) / m_stride;
      size_t previous_count = mc->used_count - previous_index;
      char *result = previous_allocated;

      if (mc->capacity_count - previous_index < count) {
        append_memory(std::max(m_total_allocated_count, count));
        memory_chunk *new_mc = &m_memory_handles.back();
        // Move the old memory to the newly allocated block
        if (previous_count > 0) {
          // Subtract the previously used memory from the old chunk's count
          mc->used_count -= previous_count;
          memcpy(new_mc->memory, previous_allocated, previous_count);
          // If the old memory only had the memory being resized,
          // free it completely.
          if (previous_allocated == mc->memory) {
            free(mc->memory);
            // Remove the second-last element of the vector
            m_memory_handles.erase(m_memory_handles.begin() + m_memory_handles.size() - 2);
          }
        }
        mc = &m_memory_handles.back();
        result = mc->memory;
        mc->used_count = count;
      } else {
        // Adjust the used count (this may mean to grow it or shrink it)
        if (count >= previous_count) {
          mc->used_count += (count - previous_count);
        } else {
          // Call the destructor on the elements no longer used
          m_dt.extended()->data_destruct_strided(m_arrmeta + arrmeta_size, previous_allocated + m_stride * count,
                                                 m_stride, previous_count - count);
          mc->used_count -= (previous_count - count);
        }
      }

      if ((m_dt.get_flags() & type_flag_zeroinit) != 0) {
        // Zero-init the new memory
        intptr_t new_count = count - (intptr_t)previous_count;
        if (new_count > 0) {
          memset(mc->memory + m_stride * previous_count, 0, m_stride * new_count);
        }
      } else {
        // TODO: Add a default data constructor to base_type
        //       as well, with a flag for it
        std::stringstream ss;
        ss << "Expected objectarray data to be zeroinit, but is not with dynd type " << m_dt;
        throw std::runtime_error(ss.str());
      }
      return result;
    }

    void finalize() { m_finalized = true; }

    void reset() {
      if (m_memory_handles.size() > 1) {
        // If there are more than one allocated memory chunks,
        // throw them all away except the last
        for (size_t i = 0, i_end = m_memory_handles.size() - 1; i != i_end; ++i) {
          memory_chunk &mc = m_memory_handles[i];
          m_dt.extended()->data_destruct_strided(m_arrmeta, mc.memory, m_stride, mc.used_count);
          free(mc.memory);
        }
        m_memory_handles.front() = m_memory_handles.back();
        m_memory_handles.resize(1);
        // Reset to zero used elements in the chunk
        memory_chunk &mc = m_memory_handles.front();
        m_dt.extended()->data_destruct_strided(m_arrmeta, mc.memory, m_stride, mc.used_count);
        mc.used_count = 0;
      }
    }

    void debug_print(std::ostream &o, const std::string &indent) {
      o << indent << "------ memory_block at " << static_cast<const void *>(this) << "\n";
      o << indent << " reference count: " << static_cast<long>(m_use_count) << "\n";
      o << " type: " << m_dt << "\n";
      o << " stride: " << m_stride << "\n";
      if (!m_finalized) {
        o << indent << " allocated count: " << m_total_allocated_count << "\n";
      } else {
        o << indent << " finalized count: " << m_total_allocated_count << "\n";
      }
      o << indent << "------" << std::endl;
    }
  };

} // namespace dynd::nd
} // namespace dynd
