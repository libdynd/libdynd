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
 * arrmeta after this structure is determined by the type
 * object.
 */
struct DYND_API array_preamble : memory_block_data {
  /**
   * type is overloaded - for builtin scalar types, it
   * simply contains the type id. If (type&~builtin_type_id_mask)
   * is 0, its a builtin type.
   */
  const ndt::base_type *type;
  uint64_t flags;
  char *data;
  intrusive_ptr<memory_block_data> ref;

  ~array_preamble();

  /** Returns true if the type is builtin */
  inline bool is_builtin_type() const
  {
    return (reinterpret_cast<uintptr_t>(type) & (~builtin_type_id_mask)) == 0;
  }

  /** Should only be called if is_builtin_type() returns true */
  inline type_id_t get_builtin_type_id() const
  {
    return static_cast<type_id_t>(reinterpret_cast<uintptr_t>(type));
  }

  inline type_id_t get_type_id() const
  {
    if (is_builtin_type()) {
      return get_builtin_type_id();
    } else {
      return type->get_type_id();
    }
  }

  /** Return a pointer to the arrmeta, immediately after the preamble */
  inline char *metadata()
  {
    return reinterpret_cast<char *>(this + 1);
  }

  /** Return a pointer to the arrmeta, immediately after the preamble */
  inline const char *metadata() const
  {
    return reinterpret_cast<const char *>(this + 1);
  }
};

/**
 * Creates a memory block for holding an nd::array (i.e. a container for nd::array arrmeta)
 *
 * The created object is uninitialized.
 */
DYND_API intrusive_ptr<memory_block_data> make_array_memory_block(size_t arrmeta_size);

/**
 * Creates a memory block for holding an nd::array (i.e. a container for nd::array arrmeta),
 * as well as storage for embedding additional POD storage such as the array data.
 *
 * The created object is uninitialized.
 */
DYND_API intrusive_ptr<memory_block_data> make_array_memory_block(size_t arrmeta_size, size_t extra_size,
                                                                  size_t extra_alignment, char **out_extra_ptr);

/**
 * Makes a shallow copy of the nd::array memory block. In the copy, only the
 * nd::array arrmeta is duplicated, all the references are the same. Any NULL
 * references are swapped to point at the original nd::array memory block, as they
 * are a signal that the data was embedded in the same memory allocation.
 */
DYND_API intrusive_ptr<memory_block_data> shallow_copy_array_memory_block(const intrusive_ptr<memory_block_data> &ndo);

DYND_API void array_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                             const std::string &indent);

} // namespace dynd
