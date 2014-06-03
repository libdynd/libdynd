//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUFFER_STORAGE_HPP_
#define _DYND__BUFFER_STORAGE_HPP_

#include <dynd/type.hpp>

namespace dynd {

template<size_t N = 128>
class buffer_storage {
    char *m_storage;
    char *m_arrmeta;
    ndt::type m_type;
    intptr_t m_stride;

    // Non-assignable
    buffer_storage& operator=(const buffer_storage&);

    void internal_allocate()
    {
        m_stride = m_type.get_data_size();
        m_storage = new char[element_count * m_stride];
        size_t metasize = m_type.is_builtin() ? 0 : m_type.extended()->get_arrmeta_size();
        if (metasize != 0) {
            try {
                m_arrmeta = NULL;
                m_arrmeta = new char[metasize];
                m_type.extended()->arrmeta_default_construct(m_arrmeta, 0, NULL);
            } catch(const std::exception&) {
                delete[] m_storage;
                delete[] m_arrmeta;
                throw;
            }
        }
    }
public:
    enum { element_count = N };

    inline buffer_storage()
        : m_storage(NULL), m_arrmeta(NULL), m_type()
    {}
    inline buffer_storage(const buffer_storage& rhs)
        : m_storage(NULL), m_arrmeta(NULL), m_type(rhs.m_type)
    {
        internal_allocate();
    }
    inline buffer_storage(const ndt::type& dt)
        : m_storage(NULL), m_arrmeta(NULL), m_type(dt)
    {
        internal_allocate();
    }
    ~buffer_storage() {
        // It's ok to call delete on a NULL pointer
        delete[] m_storage;
        if (m_arrmeta) {
            m_type.extended()->arrmeta_destruct(m_arrmeta);
            delete[] m_arrmeta;
        }
    }

    void allocate(const ndt::type& dt) {
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

    inline intptr_t get_stride() const {
        return m_stride;
    }

    inline const ndt::type& get_type() const {
        return m_type;
    }

    inline char *get_storage() const {
        return m_storage;
    }

    inline const char *get_arrmeta() const {
        return m_arrmeta;
    }

    inline void reset_arrmeta() {
        if (m_arrmeta && !m_type.is_builtin()) {
            m_type.extended()->arrmeta_reset_buffers(m_arrmeta);
        }
    }
};

} // namespace dynd

#endif // _DYND__BUFFER_STORAGE_HPP_
