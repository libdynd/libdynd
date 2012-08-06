//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "ndarray_as_py.hpp"
#include "ndarray_functions.hpp"
#include "utility_functions.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;

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
        case fixedbytes_type_id:
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

static PyObject* nested_ndarray_as_py(const dtype& d, const char *data, int ndim, const intptr_t *shape, const intptr_t *strides)
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
            pyobject_ownref item(nested_ndarray_as_py(d, data, ndim - 1, shape + 1, strides + 1));
            PyList_SET_ITEM((PyObject *)lst, i, item.release());
            data += stride;
        }
        return lst.release();
    }
}

PyObject* pydnd::ndarray_as_py(const dnd::ndarray& n)
{
    // Evaluate the ndarray, and convert strings to the Python encoding
    ndarray nvals = n.vals();

    return nested_ndarray_as_py(nvals.get_dtype(), nvals.get_readonly_originptr(),
                nvals.get_ndim(), nvals.get_shape(), nvals.get_strides());
}

