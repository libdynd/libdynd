//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
#include "dtype_functions.hpp"
#include "ndarray_functions.hpp"
#include "numpy_interop.hpp"
#include "ctypes_interop.hpp"

#include <dnd/dtypes/fixedstring_dtype.hpp>
#include <dnd/dtypes/string_dtype.hpp>

using namespace std;
using namespace dnd;
using namespace pydnd;

PyTypeObject *pydnd::WDType_Type;

void pydnd::init_w_dtype_typeobject(PyObject *type)
{
    WDType_Type = (PyTypeObject *)type;
}

dtype pydnd::deduce_dtype_from_object(PyObject* obj)
{
#if DND_NUMPY_INTEROP
    if (PyArray_Check(obj)) {
        // Numpy array
        PyArray_Descr *d = PyArray_DESCR((PyArrayObject *)obj);
        return dtype_from_numpy_dtype(d);
    } else if (PyArray_IsScalar(obj, Generic)) {
        // Numpy scalar
        return dtype_of_numpy_scalar(obj);
    }
#endif // DND_NUMPY_INTEROP
    
    if (PyBool_Check(obj)) {
        // Python bool
        return make_dtype<dnd_bool>();
    } else if (PyLong_Check(obj) || PyInt_Check(obj)) {
        // Python integer
        return make_dtype<int32_t>();
    } else if (PyFloat_Check(obj)) {
        // Python float
        return make_dtype<double>();
    } else if (PyComplex_Check(obj)) {
        // Python complex
        return make_dtype<complex<double> >();
    }

    throw std::runtime_error("could not deduce pydnd dtype from the python object");
}

/**
 * Creates a dnd::dtype out of typical Python typeobjects.
 */
static dnd::dtype make_dtype_from_pytypeobject(PyTypeObject* obj)
{
    if (obj == &PyBool_Type) {
        return make_dtype<dnd_bool>();
    } else if (obj == &PyInt_Type || obj == &PyLong_Type) {
        return make_dtype<int32_t>();
    } else if (obj == &PyFloat_Type) {
        return make_dtype<double>();
    } else if (obj == &PyComplex_Type) {
        return make_dtype<complex<double> >();
    } else if (PyObject_IsSubclass((PyObject *)obj, ctypes.PyCData_Type)) {
        // CTypes type object
        return dtype_from_ctypes_cdatatype((PyObject *)obj);
    }

    throw std::runtime_error("could not convert the given Python TypeObject into a dnd::dtype");
}

dnd::dtype pydnd::make_dtype_from_object(PyObject* obj)
{
    if (WDType_Check(obj)) {
        return ((WDType *)obj)->v;
    } else if (PyString_Check(obj)) {
        char *s = NULL;
        Py_ssize_t len = 0;
        if (PyString_AsStringAndSize(obj, &s, &len) < 0) {
            throw std::runtime_error("error processing string input to make dnd::dtype");
        }
        return dtype(string(s, len));
    } else if (PyUnicode_Check(obj)) {
        // TODO: Haven't implemented unicode yet.
        throw std::runtime_error("unicode to dnd::dtype conversion isn't implemented yet");
    } else if (PyType_Check(obj)) {
#if DND_NUMPY_INTEROP
        dtype result;
        if (dtype_from_numpy_scalar_typeobject((PyTypeObject *)obj, result) == 0) {
            return result;
        }
#endif // DND_NUMPY_INTEROP
        return make_dtype_from_pytypeobject((PyTypeObject *)obj);
    }

#if DND_NUMPY_INTEROP
    if (PyArray_DescrCheck(obj)) {
        return dtype_from_numpy_dtype((PyArray_Descr *)obj);
    }
#endif // DND_NUMPY_INTEROP

    throw std::runtime_error("could not convert the Python Object into a dnd::dtype");
}

string_encoding_t encoding_from_pyobject(PyObject *encoding_obj)
{
    string_encoding_t encoding = string_encoding_invalid;
    if (PyString_Check(encoding_obj)) {
        char *s = NULL;
        Py_ssize_t len = 0;
        if (PyString_AsStringAndSize(encoding_obj, &s, &len) < 0) {
            throw std::runtime_error("error processing string input to process string encoding");
        }
        switch (len) {
        case 5:
            switch (s[0]) {
            case 'a':
                if (strcmp(s, "ascii") == 0) {
                    encoding = string_encoding_ascii;
                }
                break;
            case 'u':
                if (strcmp(s, "utf_8") == 0) {
                    encoding = string_encoding_utf_8;
                }
                break;
            }
            break;
        case 6:
            switch (s[4]) {
            case '1':
                if (strcmp(s, "utf_16") == 0) {
                    encoding = string_encoding_utf_16;
                }
                break;
            case '3':
                if (strcmp(s, "utf_32") == 0) {
                    encoding = string_encoding_utf_32;
                }
                break;
            }
        }

        if (encoding != string_encoding_invalid) {
            return encoding;
        } else {
            stringstream ss;
            ss << "invalid input \"" << s << "\"for string encoding";
            throw std::runtime_error(ss.str());
        }

    } else if (PyUnicode_Check(encoding_obj)) {
        // TODO: Haven't implemented unicode yet.
        throw std::runtime_error("unicode isn't implemented yet for determining string encodings");
    } else {
        throw std::runtime_error("invalid input type for string encoding");
    }
}

dnd::dtype pydnd::dnd_make_fixedstring_dtype(PyObject *encoding_obj, intptr_t size)
{
    string_encoding_t encoding = encoding_from_pyobject(encoding_obj);

    return make_fixedstring_dtype(encoding, size);
}

dnd::dtype pydnd::dnd_make_string_dtype(PyObject *encoding_obj)
{
    string_encoding_t encoding = encoding_from_pyobject(encoding_obj);

    return make_string_dtype(encoding);
}