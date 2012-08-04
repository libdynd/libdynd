//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/string_dtype.hpp>
#include <dnd/memblock/external_memory_block.hpp>
#include <dnd/nodes/scalar_node.hpp>

#include "ndarray_from_py.hpp"
#include "ndarray_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;

dnd::ndarray pydnd::ndarray_from_py(PyObject *obj)
{
    // If it's a Cython w_ndarray
    if (WNDArray_Check(obj)) {
        return ((WNDArray *)obj)->v;
    }

#if DND_NUMPY_INTEROP
    if (PyArray_Check(obj)) {
        return ndarray_from_numpy_array((PyArrayObject *)obj);
    } else if (PyArray_IsScalar(obj, Generic)) {
        return ndarray_from_numpy_scalar(obj);
    }
#endif // DND_NUMPY_INTEROP

    if (PyBool_Check(obj)) {
        dnd_bool value = (obj == Py_True);
        return ndarray(make_dtype<dnd_bool>(), reinterpret_cast<const char *>(&value));
    } else if (PyInt_Check(obj)) {
        long val = PyInt_AsLong(obj);
        if (val == -1 && PyErr_Occurred()) {
            throw runtime_error("error converting int value");
        }
        return val;
    } else if (PyLong_Check(obj)) {
        long val = PyLong_AsLong(obj);
        if (val == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            PY_LONG_LONG bigval = PyLong_AsLongLong(obj);
            if (bigval == -1 && PyErr_Occurred()) {
                throw runtime_error("error converting int value");
            }
            return bigval;
        } else {
            return val;
        }
    } else if (PyFloat_Check(obj)) {
        return PyFloat_AS_DOUBLE(obj);
    } else if (PyComplex_Check(obj)) {
        return complex<double>(PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj));
    } else if (PyString_Check(obj)) { // TODO: On Python 3, PyBytes should become a dnd bytes array
        char *data = NULL;
        intptr_t len = 0;
        if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
            throw runtime_error("Error getting string data");
        }
        dtype d = make_string_dtype(string_encoding_ascii);
        const char *refs[2] = {data, data + len};
        // Python strings are immutable, so simply use the existing memory with an external memory 
        Py_INCREF(obj);
        return ndarray(make_scalar_node(d, reinterpret_cast<const char *>(&refs), read_access_flag | immutable_access_flag,
                make_external_memory_block(reinterpret_cast<void *>(obj), &py_decref_function)));
    } else if (PyUnicode_Check(obj)) {
#if Py_UNICODE_SIZE == 2
        dtype d = make_string_dtype(string_encoding_ucs_2);
#else
        dtype d = make_string_dtype(string_encoding_utf_32);
#endif
        const char *data = reinterpret_cast<const char *>(PyUnicode_AsUnicode(obj));
        const char *refs[2] = {data, data + Py_UNICODE_SIZE * PyUnicode_GetSize(obj)};
        // Python strings are immutable, so simply use the existing memory with an external memory block
        Py_INCREF(obj);
        return ndarray(make_scalar_node(d, reinterpret_cast<const char *>(&refs), read_access_flag | immutable_access_flag,
                make_external_memory_block(reinterpret_cast<void *>(obj), &py_decref_function)));
    } else {
        throw std::runtime_error("could not convert python object into a dnd::ndarray");
    }
}