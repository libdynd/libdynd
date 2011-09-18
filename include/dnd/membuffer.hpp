//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _MEMBUFFER_HPP_
#define _MEMBUFFER_HPP_

#include <dnd/dtype.hpp>

namespace dnd {

/**
 * A buffer representing a typed 1-D array of data in CPU memory.
 *
 * The eventual goal is to have different buffer types for different
 * styles of buffers. For example, A GPU buffer for arrays whose memory
 * is stored entirely on the GPU, or a global array buffer for arrays
 * whose memory is split across multiple machines in a cluster.
 *
 * TODO: Would like to add a read/write locking protocol to the buffer
 *       to enable multithreading support.
 */
class membuffer {
    /** The data type of elements in the buffer */
    dtype m_dtype;
    /** The raw memory of the buffer */
    char *m_data;
    /** The number of elements in the buffer */
    intptr_t m_size;
public:

    /**
     * Constructs the buffer with the particular data type and
     * size.
     *
     * @param d     The dtype the buffer will have. It must have itemsize
     *              greater than zero.
     * @param size  The number of elements in the buffer.
     */
    membuffer(const dtype& dt, intptr_t size);

    ~membuffer();
    
    /** The data type of elements in the buffer */
    const dtype& get_dtype() const {
        return m_dtype;
    }

    /** The number of elements in the buffer */
    intptr_t size() const {
        return m_size;
    }

    /** The raw data pointer. */
    char *data() {
        return m_data;
    }

    /** The raw data pointer (const). */
    const char *data() const {
        return m_data;
    }
};

} // namespace dnd

#endif//_MEMBUFFER_HPP_
