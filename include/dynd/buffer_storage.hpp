//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/array.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace dynd {

/**
 * Given a buffer array of type "N * T" which was
 * created by nd::empty, resets it so it can be used
 * as a buffer again.
 *
 * NOTE: If the array is not of type "N * T" and default
 *       initialized by nd::empty, undefined behavior will result.
 *
 */
inline void reset_strided_buffer_array(const nd::array &buf)
{
  const ndt::type &buf_tp = buf.get_type();
  ndt::base_type_members::flags_type flags = buf_tp.extended()->get_flags();
  if (flags & (type_flag_blockref | type_flag_zeroinit | type_flag_destructor)) {
    char *buf_arrmeta = buf.get()->metadata();
    char *buf_data = buf.data();
    buf_tp.extended()->arrmeta_reset_buffers(buf.get()->metadata());
    fixed_dim_type_arrmeta *am = reinterpret_cast<fixed_dim_type_arrmeta *>(buf_arrmeta);
    if (flags & type_flag_destructor) {
      buf_tp.extended()->data_destruct(buf_arrmeta, buf_data);
    }
    memset(buf_data, 0, am->dim_size * am->stride);
  }
}

class DYND_API buffer_storage {
  char *m_storage;
  char *m_arrmeta;
  ndt::type m_type;
  intptr_t m_stride;

  void internal_allocate()
  {
    if (m_type.get_type_id() != uninitialized_type_id) {
      m_stride = m_type.get_data_size();
      m_storage = new char[DYND_BUFFER_CHUNK_SIZE * m_stride];
      m_arrmeta = NULL;
      size_t metasize = m_type.is_builtin() ? 0 : m_type.extended()->get_arrmeta_size();
      if (metasize != 0) {
        try
        {
          m_arrmeta = new char[metasize];
          m_type.extended()->arrmeta_default_construct(m_arrmeta, true);
        }
        catch (...)
        {
          delete[] m_storage;
          delete[] m_arrmeta;
          throw;
        }
      }
    }
  }

public:
  inline buffer_storage() : m_storage(NULL), m_arrmeta(NULL), m_type()
  {
  }
  inline buffer_storage(const buffer_storage &rhs) : m_storage(NULL), m_arrmeta(NULL), m_type(rhs.m_type)
  {
    internal_allocate();
  }
  inline buffer_storage(const ndt::type &tp) : m_storage(NULL), m_arrmeta(NULL), m_type(tp)
  {
    internal_allocate();
  }
  ~buffer_storage()
  {
    if (m_storage && m_type.get_flags() & type_flag_destructor) {
      m_type.extended()->data_destruct_strided(m_arrmeta, m_storage, m_stride, DYND_BUFFER_CHUNK_SIZE);
    }
    delete[] m_storage;
    if (m_arrmeta) {
      m_type.extended()->arrmeta_destruct(m_arrmeta);
      delete[] m_arrmeta;
    }
  }

  // Assignment copies the same type
  buffer_storage &operator=(const buffer_storage &rhs)
  {
    allocate(rhs.m_type);
    return *this;
  }

  void allocate(const ndt::type &tp)
  {
    delete[] m_storage;
    m_storage = 0;
    if (m_arrmeta) {
      m_type.extended()->arrmeta_destruct(m_arrmeta);
      delete[] m_arrmeta;
      m_arrmeta = NULL;
    }
    m_type = tp;
    internal_allocate();
  }

  inline bool is_null() const
  {
    return m_storage == NULL;
  }

  inline intptr_t get_stride() const
  {
    return m_stride;
  }

  inline const ndt::type &get_type() const
  {
    return m_type;
  }

  inline char *const &get_storage() const
  {
    return m_storage;
  }

  inline const char *get_arrmeta() const
  {
    return m_arrmeta;
  }

  inline void reset_arrmeta()
  {
    if (m_arrmeta && !m_type.is_builtin()) {
      m_type.extended()->arrmeta_reset_buffers(m_arrmeta);
    }
  }
};

} // namespace dynd
