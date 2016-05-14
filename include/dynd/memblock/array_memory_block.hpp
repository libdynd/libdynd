//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/memblock/memory_block.hpp>
#include <dynd/type.hpp>
#include <dynd/types/base_memory_type.hpp>

namespace dynd {

/**
 * This structure is the start of any nd::array arrmeta. The
 * arrmeta after this structure is determined by the type
 * object.
 */
class array_preamble : public memory_block_data {
public:
  ndt::type tp;
  uint64_t flags;
  char *data;
  intrusive_ptr<memory_block_data> owner;

  array_preamble(const ndt::type &tp, size_t arrmeta_size) : tp(tp) {
    // Zero out all the arrmeta to start
    memset(reinterpret_cast<char *>(this + 1), 0, arrmeta_size);
  }

  ~array_preamble() {
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
          if (dtp.get_base_id() == memory_id) {
            dtp.extended<ndt::base_memory_type>()->data_free(data);
          }
        }
      }

      // Free the references contained in the arrmeta
      tp->arrmeta_destruct(arrmeta);
    }
  }

  /** Return a pointer to the arrmeta, immediately after the preamble */
  char *metadata() { return reinterpret_cast<char *>(this + 1); }

  /** Return a pointer to the arrmeta, immediately after the preamble */
  const char *metadata() const { return reinterpret_cast<const char *>(this + 1); }

  void debug_print(std::ostream &o, const std::string &indent) {
    o << indent << "------ memory_block at " << static_cast<const void *>(this) << "\n";
    o << indent << " reference count: " << m_use_count << "\n";
    if (!tp.is_null()) {
      o << indent << " type: " << tp << "\n";
    } else {
      o << indent << " uninitialized nd::array\n";
    }
    o << indent << "------" << std::endl;
  }

  static void *operator new(size_t size, size_t extra_size) { return ::operator new(size + extra_size); }

  static void operator delete(void *ptr) { return ::operator delete(ptr); }

  static void operator delete(void *ptr, size_t DYND_UNUSED(extra_size)) { return ::operator delete(ptr); }

  friend void intrusive_ptr_retain(array_preamble *ptr);
  friend void intrusive_ptr_release(array_preamble *ptr);
  friend long intrusive_ptr_use_count(array_preamble *ptr);
};

inline long intrusive_ptr_use_count(array_preamble *ptr) { return ptr->m_use_count; }

inline void intrusive_ptr_retain(array_preamble *ptr) { ++ptr->m_use_count; }

inline void intrusive_ptr_release(array_preamble *ptr) {
  if (--ptr->m_use_count == 0) {
    delete ptr;
  }
}

} // namespace dynd
