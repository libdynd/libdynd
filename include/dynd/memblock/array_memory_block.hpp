//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/memory_block.hpp>
#include <dynd/type.hpp>
#include <dynd/types/base_memory_type.hpp>

namespace dynd {
namespace nd {

  /**
   * This structure is the start of any nd::array arrmeta. The
   * arrmeta after this structure is determined by the type
   * object.
   */
  class DYNDT_API array_preamble : public base_memory_block {
  public:
    ndt::type tp;

  private:
    char *m_data;
    memory_block m_owner;

  public:
    uint64_t flags;

    array_preamble(const ndt::type &tp, size_t arrmeta_size, size_t data_offset, size_t data_size, uint64_t flags)
        : tp(tp), m_data(reinterpret_cast<char *>(this) + data_offset), flags(flags) {
      // Zero out all the arrmeta to start
      memset(reinterpret_cast<char *>(this + 1), 0, arrmeta_size);

      if (tp.get_flags() & type_flag_zeroinit) {
        memset(m_data, 0, data_size);
      }

      if (tp.get_flags() & type_flag_construct) {
        tp.extended()->data_construct(NULL, m_data);
      }
    }

    array_preamble(const ndt::type &tp, size_t arrmeta_size, char *data, uint64_t flags)
        : tp(tp), m_data(data), flags(flags) {
      // Zero out all the arrmeta to start
      memset(reinterpret_cast<char *>(this + 1), 0, arrmeta_size);
    }

    array_preamble(const ndt::type &tp, size_t arrmeta_size, char *data, const memory_block &owner, uint64_t flags)
        : tp(tp), m_data(data), m_owner(owner), flags(flags) {
      // Zero out all the arrmeta to start
      memset(reinterpret_cast<char *>(this + 1), 0, arrmeta_size);
    }

    ~array_preamble() {
      if (!tp.is_builtin()) {
        char *arrmeta = reinterpret_cast<char *>(this + 1);

        if (!m_owner) {
          // Call the data destructor if necessary (i.e. the nd::array owns
          // the data memory, and the type has a data destructor)
          if (!tp->is_expression() && (tp->get_flags() & type_flag_destructor) != 0) {
            tp->data_destruct(arrmeta, m_data);
          }

          // Free the ndobject data if it wasn't allocated together with the memory block
          if (!tp->is_expression()) {
            const ndt::type &dtp = tp->get_type_at_dimension(NULL, tp->get_ndim());
            if (dtp.get_base_id() == memory_id) {
              dtp.extended<ndt::base_memory_type>()->data_free(m_data);
            }
          }
        }

        // Free the references contained in the arrmeta
        tp->arrmeta_destruct(arrmeta);
      }
    }

    char *get_data() const { return m_data; }

    void set_data(char *data) { m_data = data; }

    const memory_block &get_owner() const { return m_owner; }

    void set_owner(const memory_block &owner) { m_owner = owner; }

    /** Return a pointer to the arrmeta, immediately after the preamble */
    char *metadata() { return reinterpret_cast<char *>(this + 1); }

    /** Return a pointer to the arrmeta, immediately after the preamble */
    const char *metadata() const { return reinterpret_cast<const char *>(this + 1); }

    void debug_print(std::ostream &o, const std::string &indent) {
      o << indent << "------ memory_block at " << static_cast<const void *>(this) << "\n";
      o << indent << " reference count: " << static_cast<long>(m_use_count) << "\n";
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

} // namespace dynd::nd
} // namespace dynd
