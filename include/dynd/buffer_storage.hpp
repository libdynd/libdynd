//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUFFER_STORAGE_HPP_
#define _DYND__BUFFER_STORAGE_HPP_

#include <dynd/array.hpp>
#include <dynd/types/strided_dim_type.hpp>

namespace dynd {

/**
 * Given a buffer array of type "strided * T" which was
 * created by nd::empty, resets it so it can be used
 * as a buffer again.
 *
 * NOTE: If the array is not of type "strided * T" and default
 *       initialized by nd::empty, undefined behavior will result.
 * 
 */
inline void reset_strided_buffer_array(const nd::array& buf)
{
  const ndt::type &buf_tp = buf.get_type();
  base_type_members::flags_type flags = buf_tp.extended()->get_flags();
  if (flags &
      (type_flag_blockref | type_flag_zeroinit | type_flag_destructor)) {
    char *buf_arrmeta = buf.get_ndo()->get_arrmeta();
    char *buf_data = buf.get_readwrite_originptr();
    buf_tp.extended()->arrmeta_reset_buffers(buf.get_ndo()->get_arrmeta());
    strided_dim_type_arrmeta *am =
        reinterpret_cast<strided_dim_type_arrmeta *>(buf_arrmeta);
    if (flags & type_flag_destructor) {
      buf_tp.extended()->data_destruct(buf_arrmeta, buf_data);
    }
    memset(buf_data, 0, am->dim_size * am->stride);
  }
}

class buffer_storage {
  char *m_storage;
  char *m_arrmeta;
  ndt::type m_type;
  intptr_t m_stride;

  // Non-assignable
  buffer_storage &operator=(const buffer_storage &);

  void internal_allocate()
  {
    m_stride = m_type.get_data_size();
    m_storage = new char[DYND_BUFFER_CHUNK_SIZE * m_stride];
    m_arrmeta = NULL;
    size_t metasize =
        m_type.is_builtin() ? 0 : m_type.extended()->get_arrmeta_size();
    if (metasize != 0) {
      try
      {
        m_arrmeta = new char[metasize];
        m_type.extended()->arrmeta_default_construct(m_arrmeta, 0, NULL);
      }
      catch (const std::exception &)
      {
        delete[] m_storage;
        delete[] m_arrmeta;
        throw;
      }
    }
  }

public:
  inline buffer_storage() : m_storage(NULL), m_arrmeta(NULL), m_type() {}
  inline buffer_storage(const buffer_storage &rhs)
      : m_storage(NULL), m_arrmeta(NULL), m_type(rhs.m_type)
  {
    internal_allocate();
  }
  inline buffer_storage(const ndt::type &tp)
      : m_storage(NULL), m_arrmeta(NULL), m_type(tp)
  {
    internal_allocate();
  }
  ~buffer_storage()
  {
    if (m_storage && m_type.get_flags()&type_flag_destructor) {
      m_type.extended()->data_destruct_strided(m_arrmeta, m_storage, m_stride,
                                               DYND_BUFFER_CHUNK_SIZE);
    }
    delete[] m_storage;
    if (m_arrmeta) {
      m_type.extended()->arrmeta_destruct(m_arrmeta);
      delete[] m_arrmeta;
    }
  }

  void allocate(const ndt::type &dt)
  {
    delete[] m_storage;
    m_storage = 0;
    if (m_arrmeta) {
      m_type.extended()->arrmeta_destruct(m_arrmeta);
      delete[] m_arrmeta;
      m_arrmeta = NULL;
    }
    m_type = dt;
    internal_allocate();
  }

  inline intptr_t get_stride() const { return m_stride; }

  inline const ndt::type &get_type() const { return m_type; }

  inline char *const &get_storage() const { return m_storage; }

  inline const char *get_arrmeta() const { return m_arrmeta; }

  inline void reset_arrmeta()
  {
    if (m_arrmeta && !m_type.is_builtin()) {
      m_type.extended()->arrmeta_reset_buffers(m_arrmeta);
    }
  }
};

} // namespace dynd

#endif // _DYND__BUFFER_STORAGE_HPP_
