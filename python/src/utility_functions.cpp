//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "utility_functions.hpp"

PyObject* pydnd::intptr_array_as_tuple(int size, const intptr_t *values)
{
    PyObject *result = PyTuple_New(size);
    if (result == NULL) {
        return NULL;
    }

    for (int i = 0; i < size; i++) {
        PyObject *o = PyLong_FromLongLong(values[i]);
        if (o == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, i, o);
    }

    return result;
}

