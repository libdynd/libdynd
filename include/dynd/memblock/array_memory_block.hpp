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

/**
 * This structure is the start of any nd::array arrmeta. The
 * arrmeta after this structure is determined by the m_type
 * object.
 */
struct DYND_API array_preamble {
  memory_block_data m_memblockdata;

  /**
   * m_type is overloaded - for builtin scalar types, it
   * simply contains the type id. If (m_type&~builtin_type_id_mask)
   * is 0, its a builtin type.
   */
  const ndt::base_type *m_type;
  uint64_t m_flags;
  struct {
    char *ptr;
    memory_block_data *ref;
  } data;

  /** Returns true if the type is builtin */
  inline bool is_builtin_type() const
  {
    return (reinterpret_cast<uintptr_t>(m_type) & (~builtin_type_id_mask)) == 0;
  }

  /** Should only be called if is_builtin_type() returns true */
  inline type_id_t get_builtin_type_id() const
  {
    return static_cast<type_id_t>(reinterpret_cast<uintptr_t>(m_type));
  }

  inline type_id_t get_type_id() const
  {
    if (is_builtin_type()) {
      return get_builtin_type_id();
    } else {
      return m_type->get_type_id();
    }
  }

  /** Return a pointer to the arrmeta, immediately after the preamble */
  inline char *get_arrmeta()
  {
    return reinterpret_cast<char *>(this + 1);
  }

  /** Return a pointer to the arrmeta, immediately after the preamble */
  inline const char *get_arrmeta() const
  {
    return reinterpret_cast<const char *>(this + 1);
  }
};

/**
 * Creates a memory block for holding an nd::array (i.e. a container for nd::array arrmeta)
 *
 * The created object is uninitialized.
 */
DYND_API memory_block_ptr make_array_memory_block(size_t arrmeta_size);

/**
 * Creates a memory block for holding an nd::array (i.e. a container for nd::array arrmeta),
 * as well as storage for embedding additional POD storage such as the array data.
 *
 * The created object is uninitialized.
 */
DYND_API memory_block_ptr
make_array_memory_block(size_t arrmeta_size, size_t extra_size, size_t extra_alignment, char **out_extra_ptr);

/**
 * Makes a shallow copy of the nd::array memory block. In the copy, only the
 * nd::array arrmeta is duplicated, all the references are the same. Any NULL
 * references are swapped to point at the original nd::array memory block, as they
 * are a signal that the data was embedded in the same memory allocation.
 */
DYND_API memory_block_ptr shallow_copy_array_memory_block(const memory_block_ptr &ndo);

DYND_API void array_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                             const std::string &indent);

} // namespace dynd
