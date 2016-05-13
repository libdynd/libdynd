//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <atomic>
#include <iostream>

#include <dynd/config.hpp>

namespace dynd {

/**
 * This is the data that goes at the start of every memory block, including
 * an atomic reference count. There is a fixed set of memory block types, of which 'external'
 * is presently the only extensible ones.
 */
class DYNDT_API memory_block_data {
protected:
  std::atomic_long m_use_count;

  explicit memory_block_data(long use_count) : m_use_count(use_count) {}

public:
  virtual ~memory_block_data();

  long get_use_count() const { return m_use_count; }

  /**
   * Allocates the requested amount of memory from the memory_block, returning
   * a pointer.
   *
   * Call this once per output variable.
   */
  virtual char *alloc(size_t DYND_UNUSED(count)) { throw std::runtime_error("alloc is not implemented"); }

  /**
   * Resizes the most recently allocated memory from the memory_block.
   */
  virtual char *resize(char *DYND_UNUSED(previous_allocated), size_t DYND_UNUSED(count)) {
    throw std::runtime_error("resize is not implemented");
  }

  /**
   * Finalizes the memory block so it can no longer be used to allocate more
   * memory.
   */
  virtual void finalize() { throw std::runtime_error("finalize is not implemented"); }

  /**
   * When a memory block is being used as a temporary buffer, resets it to
   * a state throwing away existing used memory. This allows the same memory
   * to be used for variable-sized data to be reused repeatedly in such
   * a temporary buffer.
   */
  virtual void reset() { throw std::runtime_error("reset is not implemented"); }

  /**
   * Does a debug dump of the memory block.
   */
  virtual void debug_print(std::ostream &o, const std::string &indent) = 0;

  void debug_print(std::ostream &o) { debug_print(o, ""); }

  friend void intrusive_ptr_retain(memory_block_data *ptr);
  friend void intrusive_ptr_release(memory_block_data *ptr);
  friend long intrusive_ptr_use_count(memory_block_data *ptr);
};

inline long intrusive_ptr_use_count(memory_block_data *ptr) { return ptr->m_use_count; }

inline void intrusive_ptr_retain(memory_block_data *ptr) { ++ptr->m_use_count; }

inline void intrusive_ptr_release(memory_block_data *ptr) {
  if (--ptr->m_use_count == 0) {
    delete ptr;
  }
}

} // namespace dynd
