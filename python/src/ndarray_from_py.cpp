//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/string_dtype.hpp>
#include <dnd/memblock/external_memory_block.hpp>
#include <dnd/nodes/scalar_node.hpp>
#include <dnd/dtype_promotion.hpp>

#include "ndarray_from_py.hpp"
#include "ndarray_functions.hpp"
#include "dtype_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;

static void deduce_pylist_shape_and_dtype(PyObject *obj, vector<intptr_t>& shape, dtype& dt, int current_axis)
{
    if (PyList_Check(obj)) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        if (shape.size() == current_axis) {
            if (dt.type_id() == void_type_id) {
                shape.push_back(size);
            } else {
                throw runtime_error("dnd:ndarray doesn't support dimensions which are sometimes scalars and sometimes arrays");
            }
        } else {
            if (shape[current_axis] != size) {
                throw runtime_error("dnd::ndarray doesn't support arrays with varying dimension sizes yet");
            }
        }
        
        for (Py_ssize_t i = 0; i < size; ++i) {
            deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, dt, current_axis + 1);
        }
    } else {
        if (shape.size() != current_axis) {
            throw runtime_error("dnd:ndarray doesn't support dimensions which are sometimes scalars and sometimes arrays");
        }

        dtype obj_dt = pydnd::deduce_dtype_from_object(obj);
        if (dt != obj_dt) {
            dt = dnd::promote_dtypes_arithmetic(obj_dt, dt);
        }
    }
}

static dnd_bool *fill_bool_ndarray_from_pylist(dnd_bool *data, PyObject *obj, const vector<intptr_t>& shape, int current_axis)
{
    if (current_axis == shape.size() - 1) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *item = PyList_GET_ITEM(obj, i);
            *data = PyObject_IsTrue(item);
            ++data;
        }
    } else {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            data = fill_bool_ndarray_from_pylist(data, PyList_GET_ITEM(obj, i), shape, current_axis + 1);
        }
    }
    return data;
}

static int32_t *fill_int32_ndarray_from_pylist(int32_t *data, PyObject *obj, const vector<intptr_t>& shape, int current_axis)
{
    if (current_axis == shape.size() - 1) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *item = PyList_GET_ITEM(obj, i);
            *data = static_cast<int32_t>(PyInt_AsLong(item));
            ++data;
        }
    } else {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            data = fill_int32_ndarray_from_pylist(data, PyList_GET_ITEM(obj, i), shape, current_axis + 1);
        }
    }
    return data;
}

static int64_t *fill_int64_ndarray_from_pylist(int64_t *data, PyObject *obj, const vector<intptr_t>& shape, int current_axis)
{
    if (current_axis == shape.size() - 1) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *item = PyList_GET_ITEM(obj, i);
            *data = PyLong_AsLongLong(item);
            ++data;
        }
    } else {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            data = fill_int64_ndarray_from_pylist(data, PyList_GET_ITEM(obj, i), shape, current_axis + 1);
        }
    }
    return data;
}

static double *fill_float64_ndarray_from_pylist(double *data, PyObject *obj, const vector<intptr_t>& shape, int current_axis)
{
    if (current_axis == shape.size() - 1) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *item = PyList_GET_ITEM(obj, i);
            *data = PyFloat_AsDouble(item);
            ++data;
        }
    } else {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            data = fill_float64_ndarray_from_pylist(data, PyList_GET_ITEM(obj, i), shape, current_axis + 1);
        }
    }
    return data;
}

static complex<double> *fill_complex_float64_ndarray_from_pylist(complex<double> *data, PyObject *obj, const vector<intptr_t>& shape, int current_axis)
{
    if (current_axis == shape.size() - 1) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *item = PyList_GET_ITEM(obj, i);
            data->real(PyComplex_RealAsDouble(item));
            data->imag(PyComplex_ImagAsDouble(item));
            ++data;
        }
    } else {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            data = fill_complex_float64_ndarray_from_pylist(data, PyList_GET_ITEM(obj, i), shape, current_axis + 1);
        }
    }
    return data;
}

static dnd::ndarray ndarray_from_pylist(PyObject *obj)
{
    // TODO: Add ability to specify access flags (e.g. immutable)
    // Do a pass through all the data to deduce its dtype and shape
    vector<intptr_t> shape;
    dtype dt;
    Py_ssize_t size = PyList_GET_SIZE(obj);
    shape.push_back(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
        deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, dt, 1);
    }

    // Create the array
    vector<int> axis_perm(shape.size());
    for (size_t i = 0, i_end = axis_perm.size(); i != i_end; ++i) {
        axis_perm[i] = i_end - i - 1;
    }
    ndarray result(dt, shape.size(), &shape[0], &axis_perm[0]);

    // Populate the array with data
    switch (dt.type_id()) {
        case bool_type_id:
            fill_bool_ndarray_from_pylist(reinterpret_cast<dnd_bool *>(result.get_readwrite_originptr()),
                            obj, shape, 0);
            break;
        case int32_type_id:
            fill_int32_ndarray_from_pylist(reinterpret_cast<int32_t *>(result.get_readwrite_originptr()),
                            obj, shape, 0);
            break;
        case int64_type_id:
            fill_int64_ndarray_from_pylist(reinterpret_cast<int64_t *>(result.get_readwrite_originptr()),
                            obj, shape, 0);
            break;
        case float64_type_id:
            fill_float64_ndarray_from_pylist(reinterpret_cast<double *>(result.get_readwrite_originptr()),
                            obj, shape, 0);
            break;
        case complex_float64_type_id:
            fill_complex_float64_ndarray_from_pylist(reinterpret_cast<complex<double> *>(result.get_readwrite_originptr()),
                            obj, shape, 0);
            break;
        default: {
            stringstream ss;
            ss << "Deduced type from Python list, " << dt << ", doesn't have a dnd::ndarray conversion function yet";
            throw runtime_error(ss.str());
        }
    }

    return result;
}

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
#if PY_VERSION_HEX < 0x03000000
    } else if (PyInt_Check(obj)) {
        long value = PyInt_AS_LONG(obj);
# if SIZEOF_LONG > SIZEOF_INT
        // Use a 32-bit int if it fits. This conversion strategy
        // is independent of sizeof(long), and is the same on 32-bit
        // and 64-bit platforms.
        if (value >= INT_MIN && value <= INT_MAX) {
            return static_cast<int>(value);
        } else {
            return value;
        }
# else
        return value;
# endif
#endif // PY_VERSION_HEX < 0x03000000
    } else if (PyLong_Check(obj)) {
        PY_LONG_LONG value = PyLong_AsLongLong(obj);
        if (value == -1 && PyErr_Occurred()) {
            throw runtime_error("error converting int value");
        }

        // Use a 32-bit int if it fits. This conversion strategy
        // is independent of sizeof(long), and is the same on 32-bit
        // and 64-bit platforms.
        if (value >= INT_MIN && value <= INT_MAX) {
            return static_cast<int>(value);
        } else {
            return value;
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
    } else if (PyList_Check(obj)) {
        return ndarray_from_pylist(obj);
    } else {
        throw std::runtime_error("could not convert python object into a dnd::ndarray");
    }
}