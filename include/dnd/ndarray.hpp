//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _NDARRAY_HPP_
#define _NDARRAY_HPP_

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/membuffer.hpp>
#include <dnd/shortvector.hpp>

#include <boost/utility/enable_if.hpp>

namespace dnd {

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
    /** Constructs an array with no buffer (NULL state) */
    ndarray();
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

    /** Swap operation (should be "noexcept" in C++11) */
    void swap(ndarray& rhs);

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
    ndarray& operator=(ndarray&& rhs) {
        if (this != &rhs) {
            m_dtype = std::move(rhs.m_dtype);
            m_ndim = rhs.m_ndim;
            m_shape = std::move(rhs.m_shape);
            m_strides = std::move(rhs.m_strides);
            m_baseoffset = rhs.m_baseoffset;
            m_buffer = std::move(rhs.m_buffer);
        }

        return *this;
    }

    int ndim() const {
        return m_ndim;
    }

    const intptr_t *shape() const {
        return m_shape.get();
    }

    const intptr_t *strides() const {
        return m_strides.get();
    }

    char *data() {
        return m_buffer->data();
    }

    const char *data() const {
        return m_buffer->data();
    }

    /** Does a value-assignment from the rhs array. */
    void vassign(const ndarray& rhs, assign_error_mode errmode = assign_error_fractional);
    /** Does a value-assignment from the rhs raw scalar */
    void vassign(const dtype& dt, const void *data, assign_error_mode errmode = assign_error_fractional);
    /** Does a value-assignment from the rhs C++ scalar. */
    template<class T>
    typename boost::enable_if<is_dtype_scalar<T>, void>::type vassign(const T& rhs,
                                                assign_error_mode errmode = assign_error_fractional) {
        vassign(make_dtype<T>(), &rhs, errmode);
    }
    void vassign(const bool& rhs, assign_error_mode errmode = assign_error_fractional) {
        vassign(dnd_bool(rhs), errmode);
    }
};

} // namespace dnd

#endif//_NDARRAY_HPP_
