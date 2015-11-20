//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/type.hpp>
#include <dynd/memblock/memory_block.hpp>
#include <dynd/types/base_memory_type.hpp>

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
  ndt::type tp;
  uint64_t flags;
  char *data;
  intrusive_ptr<memory_block_data> owner;

  ~array_preamble()
  {
    if (!tp.is_builtin()) {
      char *arrmeta = reinterpret_cast<char *>(this + 1);

      if (!owner) {
        // Call the data destructor if necessary (i.e. the nd::array owns
        // the data memory, and the type has a data destructor)
        if (!tp->is_expression() && (tp->get_flags() & type_flag_destructor) != 0) {
          tp->data_destruct(arrmeta, data);
        }

        // Free the ndobject data if it wasn't allocated together with the memory block
        if (!tp->is_expression()) {
          const ndt::type &dtp = tp->get_type_at_dimension(NULL, tp->get_ndim());
          if (dtp.get_kind() == memory_kind) {
            dtp.extended<ndt::base_memory_type>()->data_free(data);
          }
        }
      }

      // Free the references contained in the arrmeta
      tp->arrmeta_destruct(arrmeta);
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
