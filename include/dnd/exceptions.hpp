//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__EXCEPTIONS_HPP_
#define _DND__EXCEPTIONS_HPP_

#include <stdexcept>
#include <stdint.h>

#include <dnd/irange.hpp>

namespace dnd {

class dnd_exception : public std::exception {
public:
    virtual ~dnd_exception() throw() {
    }
};

/**
 * An exception for various kinds of broadcast errors.
 */
class broadcast_error : public dnd_exception {
    std::string m_what;
public:
    virtual const char* what() const throw() {
        return m_what.c_str();
    }

    /**
     * An exception for when 'src' doesn't broadcast to 'dst'
     */
    broadcast_error(int dst_ndim, const intptr_t *dst_shape,
                        int src_ndim, const intptr_t *src_shape);

    virtual ~broadcast_error() throw() {
    }
};

/**
 * An exception for an index out of bounds
 */
class too_many_indices : public dnd_exception {
    std::string m_what;
public:
    virtual const char* what() const throw() {
        return m_what.c_str();
    }

    /**
     * An exception for when too many indices are provided in
     * an indexing operation (nindex > ndim).
     */
    too_many_indices(int nindex, int ndim);

    virtual ~too_many_indices() throw() {
    }
};

class index_out_of_bounds : public dnd_exception {
    std::string m_what;
public:
    virtual const char* what() const throw() {
        return m_what.c_str();
    }

    /**
     * An exception for when 'i' isn't in the half-open range
     * [start, end).
     */
    index_out_of_bounds(intptr_t i, intptr_t start, intptr_t end);

    virtual ~index_out_of_bounds() throw() {
    }
};

/**
 * An exception for a range out of bounds
 */
class irange_out_of_bounds : public dnd_exception {
    std::string m_what;
public:
    virtual const char* what() const throw() {
        return m_what.c_str();
    }

    /**
     * An exception for when 'i' isn't in the half-open range
     * [start, end).
     */
    irange_out_of_bounds(const irange& i, intptr_t start, intptr_t end);

    virtual ~irange_out_of_bounds() throw() {
    }
};

} // namespace dnd

#endif // _DND__EXCEPTIONS_HPP_
