//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _NDARRAY_HPP_
#define _NDARRAY_HPP_

#include <dnd/dtype.hpp>
#include <dnd/membuffer.hpp>

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
    explicit shortvector(int size) {
        if (size <= N) {
            m_data = m_shortdata;
        } else {
            m_data = new T[size];
        }
    }
    shortvector(int size, const shortvector& rhs) {
        if (size <= N) {
            m_data = m_shortdata;
        } else {
            m_data = new T[size];
        }
        std::memcpy(m_data, rhs.m_data, size * sizeof(T));
    }
    /** Non-copyable */
    shortvector(const shortvector& rhs) = delete;
    /** Move constructor */
    shortvector(shortvector&& rhs)
    {
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
                m_data = rhs.m_ddata;
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

/** Typedef for vector of dimensions or strides */
typedef shortvector<intptr_t, 3> dimvector;

/**
 * This is the primary multi-dimensional array class.
 */
class ndarray {
    dtype m_dtype;
    int m_ndim;
    dimvector m_shape;
    dimvector m_strides;
    intptr_t m_baseoffset;
    std::shared_ptr<membuffer> m_buffer;

    /**
     * Private method which constructs an array from all the members. This
     * function does not validate that the strides/baseoffset stay within
     * the buffer's bounds.
     */
    ndarray(const dtype& dt, int ndim, const dimvector& shape,
            const dimvector& strides, intptr_t baseoffset,
            const std::shared_ptr<membuffer>& buffer);

public:
    /** Constructs a zero-dimensional scalar array */
    explicit ndarray(const dtype& dt);
    /** Constructs a one-dimensional array */
    ndarray(intptr_t dim0, const dtype& dt);
    /** Constructs a two-dimensional array */
    ndarray(intptr_t dim0, intptr_t dim1, const dtype& dt);
    /** Constructs a three-dimensional array */
    ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, const dtype& dt);

    /** Copy constructor */
    ndarray(const ndarray& rhs)
        : m_dtype(rhs.m_dtype), m_ndim(rhs.m_ndim),
          m_shape(rhs.m_ndim, rhs.m_shape),
          m_strides(rhs.m_ndim, rhs.m_strides),
          m_baseoffset(rhs.m_baseoffset), m_buffer(rhs.m_buffer) {}
    /** Move constructor (should just be "= default" in C++11) */
    ndarray(ndarray&& rhs)
        : m_dtype(std::move(rhs.m_dtype)), m_ndim(rhs.m_ndim),
          m_shape(std::move(rhs.m_shape)), m_strides(std::move(rhs.m_strides)),
          m_baseoffset(rhs.m_baseoffset), m_buffer(std::move(rhs.m_buffer)) {}

    /**
     * Assignment operator (should be just "= default" in C++11).
     *
     * TODO: This assignment operation copies 'rhs' with reference-like
     *       semantics. Should it instead copy the values of 'rhs' into
     *       'this'? The current way seems nicer for passing around arguments
     *       and making copies of arrays.
     */
    ndarray& operator=(const ndarray& rhs);
    /** Move assignment operator (should be just "= default" in C++11) */
    ndarray& operator=(ndarray&& rhs);

    /** Swap operation (should be "noexcept" in C++11) */
    void swap(ndarray& rhs);
};

} // namespace dnd

#endif//_NDARRAY_HPP_
