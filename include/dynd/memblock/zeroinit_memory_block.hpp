//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/memblock/base_memory_block.hpp>
#include <dynd/type.hpp>

namespace dynd {
namespace nd {

  struct zeroinit_memory_block : base_memory_block {
    size_t data_size;
    intptr_t data_alignment;
    intptr_t m_total_allocated_capacity;
    /** The malloc'd memory */
    std::vector<char *> m_memory_handles;
    /** The current malloc'd memory being doled out */
    char *m_memory_begin, *m_memory_current, *m_memory_end;

    zeroinit_memory_block(size_t data_size, intptr_t data_alignment, intptr_t initial_capacity_bytes)
        : data_size(data_size), data_alignment(data_alignment), m_total_allocated_capacity(0), m_memory_handles() {
      append_memory(initial_capacity_bytes);
    }

    ~zeroinit_memory_block() {
      for (size_t i = 0, i_end = m_memory_handles.size(); i != i_end; ++i) {
        free(m_memory_handles[i]);
      }
    }

    /**
     * Allocates some new memory from which to dole out
     * more. Adds it to the memory handles vector.
     */
    void append_memory(intptr_t capacity_bytes) {
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

    char *alloc(size_t count) {
      intptr_t size_bytes = count * data_size;

      //    cout << "allocating " << size_bytes << " of memory with alignment " << alignment << endl;
      char *begin = reinterpret_cast<char *>((reinterpret_cast<uintptr_t>(m_memory_current) + data_alignment - 1) &
                                             ~(data_alignment - 1));
      char *end = begin + size_bytes;
      if (end > m_memory_end) {
        m_total_allocated_capacity -= m_memory_end - m_memory_current;
        // Allocate memory to double the amount used so far, or the requested size, whichever is larger
        // NOTE: We're assuming malloc produces memory which has good enough alignment for anything
        append_memory(std::max(m_total_allocated_capacity, size_bytes));
        begin = m_memory_begin;
        end = begin + size_bytes;
      }

      // Indicate where to allocate the next memory
      m_memory_current = end;

      // Zero-initialize the memory
      memset(begin, 0, end - begin);

      // Return the allocated memory
      return begin;
      //    cout << "allocated at address " << (void *)begin << endl;
    }

    char *resize(char *inout_begin, size_t count) {
      intptr_t size_bytes = count * data_size;

      //    cout << "resizing memory " << (void *)*inout_begin << " / " << (void *)*inout_end << " from size " <<
      // (*inout_end - *inout_begin) << " to " << size_bytes << endl;
      //    cout << "memory state before " << (void *)m_memory_begin << " / " << (void *)m_memory_current << " /
      // " << (void *)m_memory_end << endl;
      char **inout_end = &m_memory_current;
      char *end = inout_begin + size_bytes;
      if (end <= m_memory_end) {
        // If it fits, just adjust the current allocation point
        m_memory_current = end;
        // Zero-initialize any newly allocated memory
        if (end > *inout_end) {
          memset(*inout_end, 0, end - *inout_end);
        }
        *inout_end = end;
      } else {
        // If it doesn't fit, need to copy to newly malloc'd memory
        char *old_current = inout_begin, *old_end = *inout_end;
        intptr_t old_size_bytes = *inout_end - inout_begin;
        // Allocate memory to double the amount used so far, or the requested size, whichever is larger
        // NOTE: We're assuming malloc produces memory which has good enough alignment for anything
        append_memory(std::max(m_total_allocated_capacity, size_bytes));
        memcpy(m_memory_begin, inout_begin, old_size_bytes);
        end = m_memory_begin + size_bytes;
        m_memory_current = end;
        // Zero-initialize the newly allocated memory
        memset(m_memory_begin + old_size_bytes, 0, size_bytes - old_size_bytes);

        inout_begin = m_memory_begin;
        *inout_end = end;
        m_total_allocated_capacity -= old_end - old_current;
      }
      //    cout << "memory state after " << (void *)m_memory_begin << " / " << (void *)m_memory_current << " /
      // " << (void *)m_memory_end << endl;

      return inout_begin;
    }

    void finalize() {
      if (m_memory_current < m_memory_end) {
        m_total_allocated_capacity -= m_memory_end - m_memory_current;
      }
      m_memory_begin = NULL;
      m_memory_current = NULL;
      m_memory_end = NULL;
    }

    void reset() {
      if (m_memory_handles.size() > 1) {
        // If there are more than one allocated memory chunks,
        // throw them all away except the last
        for (size_t i = 0, i_end = m_memory_handles.size() - 1; i != i_end; ++i) {
          free(m_memory_handles[i]);
        }
        m_memory_handles.front() = m_memory_handles.back();
        m_memory_handles.resize(1);
      }

      // Reset to use the whole chunk
      m_memory_current = m_memory_begin;
      m_total_allocated_capacity = m_memory_end - m_memory_begin;
    }

    void debug_print(std::ostream &o, const std::string &indent) {
      o << indent << "------ memory_block at " << static_cast<const void *>(this) << "\n";
      o << indent << " reference count: " << static_cast<long>(m_use_count) << "\n";
      if (m_memory_begin != NULL) {
        o << indent << " allocated: " << m_total_allocated_capacity << "\n";
      } else {
        o << indent << " finalized: " << m_total_allocated_capacity << "\n";
      }
      o << indent << "------" << std::endl;
    }
  };

} // namespace dynd::nd
} // namespace dynd
