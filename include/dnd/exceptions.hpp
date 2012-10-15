//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__EXCEPTIONS_HPP_
#define _DND__EXCEPTIONS_HPP_

#include <string>
#include <stdexcept>
#include <vector>

#include <dnd/irange.hpp>

namespace dynd {

// Forward declaration of ndarray class, for broadcast_error
class ndarray_node_ptr;

class dnd_exception : public std::exception {
protected:
    std::string m_message, m_what;
public:
    dnd_exception(const char *exception_name, const std::string& msg)
        : m_message(msg), m_what(std::string() + exception_name + ": " + msg)
    {
    }

    virtual const char* message() const throw() {
        return m_message.c_str();
    }
    virtual const char* what() const throw() {
        return m_what.c_str();
    }

    virtual ~dnd_exception() throw() {
    }
};

/**
 * An exception for various kinds of broadcast errors.
 */
class broadcast_error : public dnd_exception {
public:

    /**
     * An exception for when 'src' doesn't broadcast to 'dst'
     */
    broadcast_error(int dst_ndim, const intptr_t *dst_shape,
                        int src_ndim, const intptr_t *src_shape);

    /**
     * An exception for when a number of input operands can't be broadcast
     * together.
     */
    broadcast_error(int noperands, ndarray_node_ptr *operands);

    virtual ~broadcast_error() throw() {
    }
};

/**
 * An exception for an index out of bounds
 */
class too_many_indices : public dnd_exception {
public:
    /**
     * An exception for when too many indices are provided in
     * an indexing operation (nindex > ndim).
     */
    too_many_indices(int nindex, int ndim);

    virtual ~too_many_indices() throw() {
    }
};

class index_out_of_bounds : public dnd_exception {
public:
    /**
     * An exception for when 'i' isn't within bounds for
     * the specified axis of the given shape
     */
    index_out_of_bounds(intptr_t i, int axis, int ndim, const intptr_t *shape);
    index_out_of_bounds(intptr_t i, int axis, const std::vector<intptr_t>& shape);

    virtual ~index_out_of_bounds() throw() {
    }
};

class axis_out_of_bounds : public dnd_exception {
public:
    /**
     * An exception for when 'i' isn't a valid axis
     * for the number of dimensions.
     */
    axis_out_of_bounds(intptr_t i, intptr_t ndim);

    virtual ~axis_out_of_bounds() throw() {
    }
};

/**
 * An exception for a range out of bounds.
 */
class irange_out_of_bounds : public dnd_exception {
public:
    /**
     * An exception for when 'i' isn't within bounds for
     * the specified axis of the given shape
     */
    irange_out_of_bounds(const irange& i, int axis, int ndim, const intptr_t *shape);
    irange_out_of_bounds(const irange& i, int axis, const std::vector<intptr_t>& shape);

    virtual ~irange_out_of_bounds() throw() {
    }
};

/**
 * An exception for an invalid type ID.
 */
class invalid_type_id : public dnd_exception {
public:
    invalid_type_id(int type_id);

    virtual ~invalid_type_id() throw() {
    }
};

} // namespace dynd

#endif // _DND__EXCEPTIONS_HPP_
