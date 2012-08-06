//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__NDARRAY_FROM_PY_HPP_
#define _DND__NDARRAY_FROM_PY_HPP_

#include <Python.h>

#include <dnd/ndarray.hpp>

namespace pydnd {

/**
 * Converts a Python object into an ndarray using
 * the default settings.
 */
dnd::ndarray ndarray_from_py(PyObject *obj);

} // namespace pydnd

#endif // _DND__NDARRAY_FROM_PY_HPP_