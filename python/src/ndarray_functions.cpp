//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "ndarray_functions.hpp"
#include "dtype_functions.hpp"
#include "numpy_interop.hpp"

using namespace std;
using namespace dnd;

PyTypeObject *pydnd::WNDArray_Type;

void pydnd::init_w_ndarray_typeobject(PyObject *type)
{
    WNDArray_Type = (PyTypeObject *)type;
}

void pydnd::ndarray_init_from_pyobject(dnd::ndarray& n, PyObject* obj)
{
    // If it's a Cython w_ndarray
    if (WNDArray_Check(obj)) {
        n = ((WNDArray *)obj)->v;
        return;
    }

#if DND_NUMPY_INTEROP
    if (PyArray_Check(obj)) {
        n = ndarray_from_numpy_array((PyArrayObject *)obj);
        return;
    } else if (PyArray_IsScalar(obj, Generic)) {
        n = ndarray_from_numpy_scalar(obj);
        return;
    }
#endif // DND_NUMPY_INTEROP

    if (PyBool_Check(obj)) {
        n = (obj == Py_True);
    } else if (PyInt_Check(obj)) {
        long val = PyInt_AsLong(obj);
        if (val == -1 && PyErr_Occurred()) {
            throw runtime_error("error converting int value");
        }
        n = val;
    } else if (PyLong_Check(obj)) {
        long val = PyLong_AsLong(obj);
        if (val == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            PY_LONG_LONG bigval = PyLong_AsLongLong(obj);
            if (bigval == -1 && PyErr_Occurred()) {
                throw runtime_error("error converting int value");
            }
            n = bigval;
        } else {
            n = val;
        }
    } else if (PyFloat_Check(obj)) {
        n = PyFloat_AS_DOUBLE(obj);
    } else if (PyComplex_Check(obj)) {
        n = complex<double>(PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj));
    } else {
        throw std::runtime_error("could not convert python object into a dnd::ndarray");
    }
}

dnd::ndarray pydnd::ndarray_vals(const dnd::ndarray& n)
{
    return n.vals();
}
