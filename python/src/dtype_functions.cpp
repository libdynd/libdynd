#include "dtype_functions.hpp"
#include "ndarray_functions.hpp"
#include "numpy_interop.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;


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
    }

    throw std::runtime_error("could not convert the given Python TypeObject into a dnd::dtype");
}

dnd::dtype pydnd::make_dtype_from_object(PyObject* obj)
{
#if DND_NUMPY_INTEROP
    if (PyArray_DescrCheck(obj)) {
        return dtype_from_numpy_dtype((PyArray_Descr *)obj);
    }
#endif // DND_NUMPY_INTEROP

    if (PyString_Check(obj)) {
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
        return make_dtype_from_pytypeobject((PyTypeObject *)obj);
    }

    throw std::runtime_error("could not convert the Python Object into a dnd::dtype");
}
