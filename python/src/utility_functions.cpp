//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "utility_functions.hpp"

#include <dnd/exceptions.hpp>
#include <dnd/nodes/ndarray_node.hpp>

#include <Python.h>

using namespace std;
using namespace dnd;
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

static void mark_axis(PyObject *int_axis, int ndim, dnd_bool *reduce_axes)
{
    pyobject_ownref value_obj(PyNumber_Index(int_axis));
    long value = PyLong_AsLong(value_obj);
    if (value == -1 && PyErr_Occurred()) {
        throw runtime_error("error getting integer for axis argument");
    }

    if (value >= ndim || value < -ndim) {
        throw dnd::axis_out_of_bounds(value, ndim);
    } else if (value < 0) {
        value += ndim;
    }

    if (!reduce_axes[value]) {
        reduce_axes[value] = true;
    } else {
        stringstream ss;
        ss << "axis " << value << " is specified more than once";
        throw runtime_error(ss.str());
    }
}

int pydnd::pyarg_axis_argument(PyObject *axis, int ndim, dnd_bool *reduce_axes)
{
    int axis_count = 0;

    if (axis == NULL || axis == Py_None) {
        // None means use all the axes
        for (int i = 0; i < ndim; ++i) {
            reduce_axes[i] = true;
        }
        axis_count = ndim;
    } else {
        // Start with no axes marked
        for (int i = 0; i < ndim; ++i) {
            reduce_axes[i] = false;
        }
        if (PyTuple_Check(axis)) {
            // A tuple of axes
            Py_ssize_t size = PyTuple_GET_SIZE(axis);
            for (Py_ssize_t i = 0; i < size; ++i) {
                mark_axis(PyTuple_GET_ITEM(axis, i), ndim, reduce_axes);
                axis_count++;
            }
        } else  {
            // Just one axis
            mark_axis(axis, ndim, reduce_axes);
            axis_count = 1;
        }
    }

    return axis_count;
}

int pydnd::pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                const char *string0, int value0)
{
    if (obj == NULL) {
        return default_value;
    }

    if (!PyString_Check(obj)) {
        stringstream ss;
        ss << "argument " << argname << " must be a string";
        throw runtime_error(ss.str());
    }

    char *obj_str = PyString_AsString(obj);
    if (strcmp(obj_str, string0) == 0) {
        return value0;
    }

    stringstream ss;
    ss << "argument " << argname << " was given the invalid argument value \"" << obj_str << "\"";
    throw runtime_error(ss.str());
}

int pydnd::pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                const char *string0, int value0,
                const char *string1, int value1)
{
    if (obj == NULL) {
        return default_value;
    }

    if (!PyString_Check(obj)) {
        stringstream ss;
        ss << "argument " << argname << " must be a string";
        throw runtime_error(ss.str());
    }

    char *obj_str = PyString_AsString(obj);
    if (strcmp(obj_str, string0) == 0) {
        return value0;
    } else if (strcmp(obj_str, string1) == 0) {
        return value1;
    }

    stringstream ss;
    ss << "argument " << argname << " was given the invalid argument value \"" << obj_str << "\"";
    throw runtime_error(ss.str());
}

int pydnd::pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                const char *string0, int value0,
                const char *string1, int value1,
                const char *string2, int value2)
{
    if (obj == NULL) {
        return default_value;
    }

    if (!PyString_Check(obj)) {
        stringstream ss;
        ss << "argument " << argname << " must be a string";
        throw runtime_error(ss.str());
    }

    char *obj_str = PyString_AsString(obj);
    if (strcmp(obj_str, string0) == 0) {
        return value0;
    } else if (strcmp(obj_str, string1) == 0) {
        return value1;
    } else if (strcmp(obj_str, string2) == 0) {
        return value2;
    }

    stringstream ss;
    ss << "argument " << argname << " was given the invalid argument value \"" << obj_str << "\"";
    throw runtime_error(ss.str());
}

bool pydnd::pyarg_bool(PyObject *obj, const char *argname, bool default_value)
{
    if (obj == NULL) {
        return default_value;
    }

    if (obj == Py_False) {
        return false;
    } else if (obj = Py_True) {
        return true;
    } else {
        stringstream ss;
        ss << "argument " << argname << " must be a boolean True or False";
        throw runtime_error(ss.str());
    }
}

uint32_t pydnd::pyarg_access_flags(PyObject* obj)
{
    pyobject_ownref iterator(PyObject_GetIter(obj));
    PyObject *item_raw;

    uint32_t result = 0;

    while (item_raw = PyIter_Next(iterator)) {
        pyobject_ownref item(item_raw);
        result |= (uint32_t)pyarg_strings_to_int(item, "access_flags", 0,
                    "read", read_access_flag,
                    "write", write_access_flag,
                    "immutable", immutable_access_flag);
    }

    if (PyErr_Occurred()) {
        throw runtime_error("propagating exception...");
    }

    return result;
}
