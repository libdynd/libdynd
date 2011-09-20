//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _SHORTVECTOR_HPP_
#define _SHORTVECTOR_HPP_

// For std::memcpy
#include <cstring>

namespace dnd {

/**
 * This is a helper class for storing vectors that are
 * usually short. It doesn't store the actual length of the
 * array, however. When the vector is of length N or shorter,
 * no memory is allocated on the heap, otherwise it is.
 *
 * The purpose of this is to allow arrays with fewer dimensions
 * to be manipulated without allocating memory on the heap
 * for the shape and strides, but without sacrificing full generality
 * in the number of dimensions.
 */
template<class T, int N>
class shortvector {
    T *m_data;
    T m_shortdata[N];
public:
    /** Construct the shortvector with a specified size */
    explicit shortvector(int size) {
        if (size <= N) {
            m_data = m_shortdata;
        } else {
            m_data = new T[size];
        }
    }
    /** Construct the shortvector with a specified size and initial data */
    shortvector(int size, const shortvector& rhs) {
        // Could use C++11 delegating constructor for the first part
        if (size <= N) {
            m_data = m_shortdata;
        } else {
            m_data = new T[size];
        }
        std::memcpy(m_data, rhs.m_data, size * sizeof(T));
    }
    /** Construct the shortvector with a specified size and initial data */
    shortvector(int size, const T* data) {
        // Could use C++11 delegating constructor for the first part
        if (size <= N) {
            m_data = m_shortdata;
        } else {
            m_data = new T[size];
        }
        std::memcpy(m_data, data, size * sizeof(T));
    }
    /** Non-copyable */
    shortvector(const shortvector& rhs) = delete;
    /** Move constructor */
    shortvector(shortvector&& rhs) {
        if (rhs.m_data == rhs.m_shortdata) {
            // In the short case, copy the full shortdata vector
            std::memcpy(m_shortdata, rhs.m_shortdata, N * sizeof(T));
            m_data = m_shortdata;
        } else {
            // In the long case, move the long allocated pointer
            m_data = rhs.m_data;
            rhs.m_data = rhs.m_shortdata;
        }
    }
    /** Non-copyable (this object wouldn't know how much to allocate anyway) */
    shortvector& operator=(const shortvector& rhs) = delete;
    /** Move assignment operator */
    shortvector& operator=(shortvector&& rhs) {
        if (this != &rhs) {
            if (m_data != m_shortdata) {
                delete[] m_data;
            }
            if (rhs.m_data == rhs.m_shortdata) {
                // In the short case, copy the full shortdata vector
                std::memcpy(m_shortdata, rhs.m_shortdata, N * sizeof(T));
                m_data = m_shortdata;
            } else {
                // In the long case, move the long allocated pointer
                m_data = rhs.m_data;
                rhs.m_data = rhs.m_shortdata;
            }
        }
        return *this;
    }

    /** Destructor */
    ~shortvector() {
        if (m_data != m_shortdata) {
            delete[] m_data;
        }
    }

    void swap(shortvector& rhs) {
        // Start by swapping the pointers
        std::swap(m_data, rhs.m_data);
        // Copy the shortdata if necessary
        if (m_data == rhs.m_shortdata) {
            // The rhs data was pointing to shortdata
            m_data = m_shortdata;
            if (rhs.m_data == m_shortdata) {
                // Both data's were pointing to their shortdata
                T tmp[N];
                rhs.m_data = rhs.m_shortdata;
                std::memcpy(tmp, m_shortdata, N * sizeof(T));
                std::memcpy(m_shortdata, rhs.m_shortdata, N * sizeof(T));
                std::memcpy(rhs.m_shortdata, tmp, N * sizeof(T));
            } else {
                // Just the rhs data was pointing to shortdata
                std::memcpy(m_shortdata, rhs.m_shortdata, N * sizeof(T));
            }
        } else if (rhs.m_data == m_shortdata) {
            // Just this data was pointing to shortdata
            rhs.m_data = rhs.m_shortdata;
            std::memcpy(rhs.m_shortdata, m_shortdata, N * sizeof(T));
        }
    }

    /** Gets the const data pointer */
    const T* get() const {
        return m_data;
    }
    /** Gets the non-const data pointer */
    T* get() {
        return m_data;
    }

    /** Const indexing operator */
    const T& operator[](int i) const {
        return m_data[i];
    }
    /** Non-const indexing operator */
    T& operator[](int i) {
        return m_data[i];
    }
};

} // namespace dnd

#endif // _SHORTVECTOR_HPP_
