//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some wrapping functions to
// access various ndarray parameters
//

#ifndef _DND__NDARRAY_FUNCTIONS_HPP_
#define _DND__NDARRAY_FUNCTIONS_HPP_

#include <stdint.h>
#include <sstream>

#include <dnd/ndarray.hpp>

#include "Python.h"

namespace pydnd {

void ndarray_init(dnd::ndarray& n, PyObject* obj);
dnd::ndarray ndarray_vals(const dnd::ndarray& n);

inline dnd::ndarray ndarray_add(const dnd::ndarray& lhs, const dnd::ndarray& rhs)
{
    return lhs + rhs;
}

inline dnd::ndarray ndarray_subtract(const dnd::ndarray& lhs, const dnd::ndarray& rhs)
{
    return lhs - rhs;
}

inline dnd::ndarray ndarray_multiply(const dnd::ndarray& lhs, const dnd::ndarray& rhs)
{
    return lhs * rhs;
}

inline dnd::ndarray ndarray_divide(const dnd::ndarray& lhs, const dnd::ndarray& rhs)
{
    return lhs / rhs;
}

inline std::string ndarray_str(const dnd::ndarray& n)
{
    std::stringstream ss;
    ss << n;
    return ss.str();
}

inline std::string ndarray_repr(const dnd::ndarray& n)
{
    std::stringstream ss;
    ss << n;
    return ss.str();
}

inline std::string ndarray_debug_dump(const dnd::ndarray& n)
{
    std::stringstream ss;
    n.debug_dump(ss);
    return ss.str();
}

PyObject* intptr_array_as_tuple(int size, const intptr_t *array);

} // namespace pydnd

#endif // _DND__NDARRAY_FUNCTIONS_HPP_