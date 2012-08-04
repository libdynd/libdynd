//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__NDARRAY_AS_PY_HPP_
#define _DND__NDARRAY_AS_PY_HPP_

#include <Python.h>

#include <dnd/ndarray.hpp>

namespace pydnd {

/**
 * Converts an ndarray into a Python object
 * using the default settings.
 */
PyObject *ndarray_as_py(const dnd::ndarray& n);

} // namespace pydnd

#endif // _DND__NDARRAY_AS_PY_HPP_