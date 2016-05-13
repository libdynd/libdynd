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
 * These are all the types of memory blocks supported by the dnd library.
 */
enum memory_block_type_t {
  /** A dynd array containing the arrmeta specified by the type */
  array_memory_block_type,
  /** Memory from outside the dnd library */
  external_memory_block_type,
  /** For when the data is POD and its size is fully known ahead of time */
  fixed_size_pod_memory_block_type,
  /** For when the data is POD, and the amount of memory needs to grow */
  pod_memory_block_type,
  /** Like pod_memory_block_type, but with zero-initialization */
  zeroinit_memory_block_type,
  /**
   * For when the data is object (requires destruction),
   * and the amount of memory needs to grow */
  objectarray_memory_block_type,
  /** Wraps memory mapped files */
  memmap_memory_block_type
};

DYNDT_API std::ostream &operator<<(std::ostream &o, memory_block_type_t mbt);

/**
 * This is the data that goes at the start of every memory block, including
 * an atomic reference count and a memory_block_type_t. There is a fixed set
 * of memory block types, of which 'external' is presently the only
 * extensible ones.
 */
class DYNDT_API memory_block_data {
protected:
  std::atomic_long m_use_count;

  explicit memory_block_data(long use_count, memory_block_type_t type) : m_use_count(use_count), m_type(type) {
    // std::cout << "memblock " << (void *)this << " cre: " << this->m_use_count << std::endl;
  }

public:
  /** A memory_block_type_t enum value */
  uint32_t m_type;

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

/**
 * Does a debug dump of the memory block.
 */
DYNDT_API void memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                        const std::string &indent = "");

} // namespace dynd
