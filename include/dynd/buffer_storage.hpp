//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUFFER_STORAGE_HPP_
#define _DYND__BUFFER_STORAGE_HPP_

#include <dynd/dtype.hpp>

namespace dynd {

template<size_t N = 128>
class buffer_storage {
    char *m_storage;
    char *m_metadata;
    dtype m_dtype;
    intptr_t m_stride;

    // Non-assignable
    buffer_storage& operator=(const buffer_storage&);

    void internal_allocate()
    {
        m_stride = m_dtype.get_data_size();
        m_storage = new char[element_count * m_stride];
        size_t metasize = m_dtype.is_builtin() ? 0 : m_dtype.extended()->get_metadata_size();
        if (metasize != 0) {
            try {
                m_metadata = NULL;
                m_metadata = new char[metasize];
                m_dtype.extended()->metadata_default_construct(m_metadata, 0, NULL);
            } catch(const std::exception&) {
                delete[] m_storage;
                delete[] m_metadata;
                throw;
            }
        }
    }
public:
    enum { element_count = N };

    inline buffer_storage()
        : m_storage(NULL), m_metadata(NULL), m_dtype()
    {}
    inline buffer_storage(const buffer_storage& rhs)
        : m_storage(NULL), m_metadata(NULL), m_dtype(rhs.m_dtype)
    {
        internal_allocate();
    }
    inline buffer_storage(const dtype& dt)
        : m_storage(NULL), m_metadata(NULL), m_dtype(dt)
    {
        internal_allocate();
    }
    ~buffer_storage() {
        // It's ok to call delete on a NULL pointer
        delete[] m_storage;
        if (m_metadata) {
            m_dtype.extended()->metadata_destruct(m_metadata);
            delete[] m_metadata;
        }
    }

    void allocate(const dtype& dt) {
        delete[] m_storage;
        m_storage = 0;
        if (m_metadata) {
            m_dtype.extended()->metadata_destruct(m_metadata);
            delete[] m_metadata;
            m_metadata = NULL;
        }
        m_dtype = dt;
        internal_allocate();
    }

    inline intptr_t get_stride() const {
        return m_stride;
    }

    inline const dtype& get_dtype() const {
        return m_dtype;
    }

    inline char *get_storage() const {
        return m_storage;
    }

    inline const char *get_metadata() const {
        return m_metadata;
    }

    inline void reset_metadata() {
        if (m_metadata && !m_dtype.is_builtin()) {
            m_dtype.extended()->metadata_reset_buffers(m_metadata);
        }
    }
};

} // namespace dynd

#endif // _DYND__BUFFER_STORAGE_HPP_
