//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "ndarray_functions.hpp"
#include "dtype_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

#include <dnd/dtypes/string_dtype.hpp>
#include <dnd/memblock/external_memory_block.hpp>
#include <dnd/nodes/scalar_node.hpp>
#include <dnd/ndarray_arange.hpp>
#include <dnd/dtype_promotion.hpp>

using namespace std;
using namespace dnd;
using namespace pydnd;

PyTypeObject *pydnd::WNDArray_Type;

void pydnd::init_w_ndarray_typeobject(PyObject *type)
{
    WNDArray_Type = (PyTypeObject *)type;
}

static void python_decref(void *obj)
{
    PyObject *pyobj = reinterpret_cast<PyObject *>(obj);
    Py_DECREF(pyobj);
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
        dnd_bool value = (obj == Py_True);
        n = ndarray(make_dtype<dnd_bool>(), reinterpret_cast<const char *>(&value));
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
        n = ndarray(make_scalar_node(d, reinterpret_cast<const char *>(&refs), read_access_flag | immutable_access_flag,
                make_external_memory_block(reinterpret_cast<void *>(obj), &python_decref)));
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
        n = ndarray(make_scalar_node(d, reinterpret_cast<const char *>(&refs), read_access_flag | immutable_access_flag,
                make_external_memory_block(reinterpret_cast<void *>(obj), &python_decref)));
    } else {
        throw std::runtime_error("could not convert python object into a dnd::ndarray");
    }
}

dnd::ndarray pydnd::ndarray_vals(const dnd::ndarray& n)
{
    return n.vals();
}

static PyObject* element_as_pyobject(const dtype& d, const char *data)
{
    switch (d.type_id()) {
        case bool_type_id:
            if (*(const dnd_bool *)data) {
                Py_INCREF(Py_True);
                return Py_True;
            } else {
                Py_INCREF(Py_False);
                return Py_False;
            }
        case int8_type_id:
            return PyInt_FromLong(*(const int8_t *)data);
        case int16_type_id:
            return PyInt_FromLong(*(const int16_t *)data);
        case int32_type_id:
            return PyInt_FromLong(*(const int32_t *)data);
        case int64_type_id:
            return PyLong_FromLongLong(*(const int64_t *)data);
        case uint8_type_id:
            return PyInt_FromLong(*(const uint8_t *)data);
        case uint16_type_id:
            return PyInt_FromLong(*(const uint16_t *)data);
        case uint32_type_id:
            return PyInt_FromLong(*(const uint32_t *)data);
        case uint64_type_id:
            return PyLong_FromUnsignedLongLong(*(const uint64_t *)data);
        case float32_type_id:
            return PyFloat_FromDouble(*(const float *)data);
        case float64_type_id:
            return PyFloat_FromDouble(*(const double *)data);
        case complex_float32_type_id:
            return PyComplex_FromDoubles(*(const float *)data, *((const float *)data + 1));
        case complex_float64_type_id:
            return PyComplex_FromDoubles(*(const double *)data, *((const double *)data + 1));
        case bytes_type_id:
            return PyBytes_FromStringAndSize(data, d.element_size());
        case fixedstring_type_id: {
            switch (d.string_encoding()) {
                case string_encoding_ascii:
                    return PyUnicode_DecodeASCII(data, strnlen(data, d.element_size()), NULL);
                case string_encoding_utf_8:
                    return PyUnicode_DecodeUTF8(data, strnlen(data, d.element_size()), NULL);
                case string_encoding_ucs_2:
                case string_encoding_utf_16: {
                    // Get the null-terminated string length
                    const uint16_t *udata = (const uint16_t *)data;
                    const uint16_t *udata_end = udata;
                    intptr_t size = d.element_size() / sizeof(uint16_t);
                    while (size > 0 && *udata_end != 0) {
                        --size;
                        ++udata_end;
                    }
                    return PyUnicode_DecodeUTF16(data, sizeof(uint16_t) * (udata_end - udata), NULL, NULL);
                }
                case string_encoding_utf_32: {
                    // Get the null-terminated string length
                    const uint32_t *udata = (const uint32_t *)data;
                    const uint32_t *udata_end = udata;
                    intptr_t size = d.element_size() / sizeof(uint32_t);
                    while (size > 0 && *udata_end != 0) {
                        --size;
                        ++udata_end;
                    }
                    return PyUnicode_DecodeUTF32(data, sizeof(uint32_t) * (udata_end - udata), NULL, NULL);
                }
                default:
                    throw runtime_error("Unrecognized dnd::ndarray string encoding");
            }
        }
        case string_type_id: {
            const char * const *refs = reinterpret_cast<const char * const *>(data);
            switch (d.string_encoding()) {
                case string_encoding_ascii:
                    return PyUnicode_DecodeASCII(refs[0], refs[1] - refs[0], NULL);
                case string_encoding_utf_8:
                    return PyUnicode_DecodeUTF8(refs[0], refs[1] - refs[0], NULL);
                case string_encoding_ucs_2:
                case string_encoding_utf_16:
                    return PyUnicode_DecodeUTF16(refs[0], refs[1] - refs[0], NULL, NULL);
                case string_encoding_utf_32:
                    return PyUnicode_DecodeUTF32(refs[0], refs[1] - refs[0], NULL, NULL);
                default:
                    throw runtime_error("Unrecognized dnd::ndarray string encoding");
            }
        }
        default: {
            stringstream ss;
            ss << "Cannot convert dnd::ndarray with dtype " << d << " into python object";
            throw runtime_error(ss.str());
        }
    }
}

static PyObject* nested_ndarray_as_pyobject(const dtype& d, const char *data, int ndim, const intptr_t *shape, const intptr_t *strides)
{
    if (ndim == 0) {
        return element_as_pyobject(d, data);
    } else if (ndim == 1) {
        pyobject_ownref lst(PyList_New(shape[0]));
        for (intptr_t i = 0; i < shape[0]; ++i) {
            pyobject_ownref item(element_as_pyobject(d, data));
            PyList_SET_ITEM((PyObject *)lst, i, item.release());
            data += strides[0];
        }
        return lst.release();
    } else {
        pyobject_ownref lst(PyList_New(shape[0]));
        intptr_t size = *shape;
        intptr_t stride = *strides;
        for (intptr_t i = 0; i < size; ++i) {
            pyobject_ownref item(nested_ndarray_as_pyobject(d, data, ndim - 1, shape + 1, strides + 1));
            PyList_SET_ITEM((PyObject *)lst, i, item.release());
            data += stride;
        }
        return lst.release();
    }
}

PyObject* pydnd::ndarray_as_pyobject(const dnd::ndarray& n)
{
    ndarray nvals;

    // Evaluate the ndarray, and convert strings to the Python encoding
    nvals = n.vals();

    return nested_ndarray_as_pyobject(n.get_dtype(), n.get_readonly_originptr(),
                n.get_ndim(), n.get_shape(), n.get_strides());
}

static irange pyobject_as_irange(PyObject *index)
{
    if (PySlice_Check(index)) {
        irange result;
        PySliceObject *slice = (PySliceObject *)index;
        if (slice->start != Py_None) {
            result.set_start(pyobject_as_index(slice->start));
        }
        if (slice->stop != Py_None) {
            result.set_finish(pyobject_as_index(slice->stop));
        }
        if (slice->step != Py_None) {
            result.set_step(pyobject_as_index(slice->step));
        }
        return result;
    } else {
        return irange(pyobject_as_index(index));
    }
}

dnd::ndarray pydnd::ndarray_getitem(const dnd::ndarray& n, PyObject *subscript)
{
    // Convert the pyobject into an array of iranges
    intptr_t size;
    shortvector<irange> indices;
    if (!PyTuple_Check(subscript)) {
        // A single subscript
        size = 1;
        indices.init(1);
        indices[0] = pyobject_as_irange(subscript);
    } else {
        size = PyTuple_GET_SIZE(subscript);
        // Tuple of subscripts
        indices.init(size);
        for (Py_ssize_t i = 0; i < size; ++i) {
            indices[i] = pyobject_as_irange(PyTuple_GET_ITEM(subscript, i));
        }
    }

    // Do an indexing operation
    return n.index(size, indices.get());
}

ndarray pydnd::ndarray_arange(PyObject *start, PyObject *stop, PyObject *step)
{
    ndarray start_nd, stop_nd, step_nd;
    if (start != Py_None) {
        ndarray_init_from_pyobject(start_nd, start);
    } else {
        start_nd = 0;
    }
    ndarray_init_from_pyobject(stop_nd, stop);
    if (step != Py_None) {
        ndarray_init_from_pyobject(step_nd, step);
    } else {
        step_nd = 1;
    }
    
    dtype dt = promote_dtypes_arithmetic(start_nd.get_dtype(),
            promote_dtypes_arithmetic(stop_nd.get_dtype(), step_nd.get_dtype()));
    
    start_nd = start_nd.as_dtype(dt, assign_error_none).vals();
    stop_nd = stop_nd.as_dtype(dt, assign_error_none).vals();
    step_nd = step_nd.as_dtype(dt, assign_error_none).vals();

    if (start_nd.get_ndim() > 0 || stop_nd.get_ndim() > 0 || step_nd.get_ndim()) {
        throw runtime_error("dnd::arange should only be called with scalar parameters");
    }

    return arange(dt, start_nd.get_readonly_originptr(),
            stop_nd.get_readonly_originptr(),
            step_nd.get_readonly_originptr());
}

dnd::ndarray pydnd::ndarray_linspace(PyObject *start, PyObject *stop, PyObject *count)
{
    ndarray start_nd, stop_nd;
    intptr_t count_val = pyobject_as_index(count);
    ndarray_init_from_pyobject(start_nd, start);
    ndarray_init_from_pyobject(stop_nd, stop);
    dtype dt = promote_dtypes_arithmetic(start_nd.get_dtype(), stop_nd.get_dtype());
    // Make sure it's at least floating point
    if (dt.kind() == bool_kind || dt.kind() == int_kind || dt.kind() == uint_kind) {
        dt = make_dtype<double>();
    }
    start_nd = start_nd.as_dtype(dt, assign_error_none).vals();
    stop_nd = stop_nd.as_dtype(dt, assign_error_none).vals();

    if (start_nd.get_ndim() > 0 || stop_nd.get_ndim() > 0) {
        throw runtime_error("dnd::linspace should only be called with scalar parameters");
    }

    return linspace(dt, start_nd.get_readonly_originptr(), stop_nd.get_readonly_originptr(), count_val);
}
