//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "utility_functions.hpp"

#include <Python.h>

using namespace std;
using namespace pydnd;

intptr_t pydnd::pyobject_as_index(PyObject *index)
{
    pyobject_ownref start_obj(PyNumber_Index(index));
    intptr_t result = PyLong_AsLongLong(start_obj);
    if (result == -1 && PyErr_Occurred()) {
        throw runtime_error("error getting index integer"); // TODO: propagate Python exception
    }
    return result;
}

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

