//
// Copyright (C) 2012 Continuum Analytics
// All rights reserved.
//
#ifndef _DND__BUFFER_STORAGE_HPP_
#define _DND__BUFFER_STORAGE_HPP_

#include <dnd/dtype.hpp>

namespace dnd {

class buffer_storage {
    char *m_storage;
    intptr_t m_element_count;

    // Non-copyable
    buffer_storage(const buffer_storage&);
    buffer_storage& operator=(const buffer_storage&);
public:
    buffer_storage(const dtype& element_dtype, intptr_t max_element_count, intptr_t max_byte_count = 16384) {
        m_element_count = max_byte_count / element_dtype.itemsize();
        if (m_element_count > max_element_count) {
            m_element_count = max_element_count;
        }
        if (m_element_count == 0) {
            m_element_count = 1;
        }
        m_storage = new char[m_element_count * element_dtype.itemsize()];
    }
    ~buffer_storage() {
        delete[] m_storage;
    }

    char *storage() const {
        return m_storage;
    }

    intptr_t element_count() const {
        return m_element_count;
    }
};

} // namespace dnd

#endif // _DND__BUFFER_STORAGE_HPP_