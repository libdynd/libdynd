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

  enum array_access_flags {
    /** If an array is readable */
    read_access_flag = 0x01,
    /** If an array is writable */
    write_access_flag = 0x02,
    /** If an array will not be written to by anyone else either */
    immutable_access_flag = 0x04
  };

  // Some additional access flags combinations for convenience
  enum {
    readwrite_access_flags = read_access_flag | write_access_flag,
    default_access_flags = read_access_flag | write_access_flag,
  };

  /**
   * This structure is the start of any nd::array arrmeta. The
   * arrmeta after this structure is determined by the type
   * object.
   */
  class DYNDT_API array_preamble : public base_memory_block {
    ndt::type m_tp;
    char *m_data;
    memory_block m_owner;
    uint64_t m_flags;

  public:
    array_preamble(const ndt::type &tp, size_t data_offset, size_t data_size, uint64_t flags)
        : m_tp(tp), m_data(reinterpret_cast<char *>(this) + data_offset), m_flags(flags) {
      // Zero out all the arrmeta to start
      memset(reinterpret_cast<char *>(this + 1), 0, m_tp.get_arrmeta_size());

      if (m_tp.get_flags() & type_flag_zeroinit) {
        memset(m_data, 0, data_size);
      }

      if (tp.get_flags() & type_flag_construct) {
        m_tp.extended()->data_construct(NULL, m_data);
      }
    }

    array_preamble(const ndt::type &tp, char *data, uint64_t flags) : m_tp(tp), m_data(data), m_flags(flags) {
      // Zero out all the arrmeta to start
      memset(reinterpret_cast<char *>(this + 1), 0, m_tp.get_arrmeta_size());
    }

    array_preamble(const ndt::type &tp, char *data, const memory_block &owner, uint64_t flags)
        : m_tp(tp), m_data(data), m_owner(owner), m_flags(flags) {
      // Zero out all the arrmeta to start
      memset(reinterpret_cast<char *>(this + 1), 0, m_tp.get_arrmeta_size());
    }

    ~array_preamble() {
      if (!m_tp.is_builtin()) {
        char *arrmeta = reinterpret_cast<char *>(this + 1);

        if (!m_owner) {
          // Call the data destructor if necessary (i.e. the nd::array owns
          // the data memory, and the type has a data destructor)
          if (!m_tp->is_expression() && (m_tp->get_flags() & type_flag_destructor) != 0) {
            m_tp->data_destruct(arrmeta, m_data);
          }

          // Free the ndobject data if it wasn't allocated together with the memory block
          if (!m_tp->is_expression()) {
            const ndt::type &dtp = m_tp->get_type_at_dimension(NULL, m_tp->get_ndim());
            if (dtp.get_base_id() == memory_id) {
              dtp.extended<ndt::base_memory_type>()->data_free(m_data);
            }
          }
        }

        // Free the references contained in the arrmeta
        m_tp->arrmeta_destruct(arrmeta);
      }
    }

    char *get_data() const { return m_data; }

    /** Return a pointer to the arrmeta, immediately after the preamble */
    char *metadata() const { return const_cast<char *>(reinterpret_cast<const char *>(this + 1)); }

    /** Return a pointer to the arrmeta, immediately after the preamble */
    //    const char *metadata() const { return reinterpret_cast<const char *>(this + 1); }

    void debug_print(std::ostream &o, const std::string &indent) {
      o << indent << "------ memory_block at " << static_cast<const void *>(this) << "\n";
      o << indent << " reference count: " << static_cast<long>(m_use_count) << "\n";
      if (!m_tp.is_null()) {
        o << indent << " type: " << m_tp << "\n";
      } else {
        o << indent << " uninitialized nd::array\n";
      }
      o << indent << "------" << std::endl;
    }

    static void *operator new(size_t size, size_t extra_size) { return ::operator new(size + extra_size); }

    static void operator delete(void *ptr) { return ::operator delete(ptr); }

    static void operator delete(void *ptr, size_t DYND_UNUSED(extra_size)) { return ::operator delete(ptr); }

    friend class buffer;

    friend void intrusive_ptr_retain(const array_preamble *ptr);
    friend void intrusive_ptr_release(const array_preamble *ptr);
    friend long intrusive_ptr_use_count(const array_preamble *ptr);
  };

  inline long intrusive_ptr_use_count(const array_preamble *ptr) { return ptr->m_use_count; }

  inline void intrusive_ptr_retain(const array_preamble *ptr) { ++ptr->m_use_count; }

  inline void intrusive_ptr_release(const array_preamble *ptr) {
    if (--ptr->m_use_count == 0) {
      delete ptr;
    }
  }

} // namespace dynd::nd
} // namespace dynd
