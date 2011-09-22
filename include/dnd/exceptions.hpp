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

} // namespace dnd

#endif // _DND__EXCEPTIONS_HPP_
