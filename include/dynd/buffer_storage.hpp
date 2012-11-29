//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUFFER_STORAGE_HPP_
#define _DYND__BUFFER_STORAGE_HPP_

#include <dynd/dtype.hpp>

namespace dynd {

class buffer_storage {
    char *m_storage;
    intptr_t m_element_size, m_element_count;

    // Non-assignable
    buffer_storage& operator=(const buffer_storage&);

    void internal_allocate(intptr_t element_size, intptr_t max_element_count, intptr_t max_byte_count)
    {
        m_element_count = max_byte_count / element_size;
        if (m_element_count > max_element_count) {
            m_element_count = max_element_count;
        }
        if (m_element_count == 0) {
            m_element_count = 1;
        }
        m_element_size = element_size;
        m_storage = new char[m_element_count * element_size];
    }
public:
    buffer_storage() {
        m_storage = 0;
    }
    buffer_storage(const buffer_storage& rhs)
        : m_storage(new char[rhs.m_element_count * rhs.m_element_size]),
            m_element_size(rhs.m_element_size), m_element_count(rhs.m_element_count)
    {}
    buffer_storage(intptr_t element_size, intptr_t max_element_count = 32768, intptr_t max_byte_count = 32768) {
        internal_allocate(element_size, max_element_count, max_byte_count);
    }
    ~buffer_storage() {
        // It's ok to call delete on a NULL pointer
        delete[] m_storage;
    }

    void allocate(intptr_t element_size, intptr_t max_element_count = 32768, intptr_t max_byte_count = 32768) {
        delete[] m_storage;
        m_storage = 0;
        internal_allocate(element_size, max_element_count, max_byte_count);
    }

    char *storage() const {
        return m_storage;
    }

    intptr_t get_element_size() const {
        return m_element_size;
    }

    intptr_t get_element_count() const {
        return m_element_count;
    }
};

} // namespace dynd

#endif // _DYND__BUFFER_STORAGE_HPP_
