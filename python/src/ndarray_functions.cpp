#include "ndarray_functions.hpp"
#include "dtype_functions.hpp"

using namespace std;
using namespace dnd;

template<class T>
static void py_decref_function(T* obj)
{
    Py_DECREF(obj);
}

inline ndarray from_numpy_array(PyArrayObject* obj)
{
    // Get the dtype of the array
    dtype d = pydnd::dtype_from_numpy_dtype(PyArray_DESCR(obj));
    // Get a shared pointer that tracks buffer ownership
    PyObject *base = PyArray_BASE(obj);
    dnd::shared_ptr<void> bufowner;
    if (base == NULL || (PyArray_FLAGS(obj)&NPY_ARRAY_UPDATEIFCOPY) != 0) {
        Py_INCREF(obj);
        bufowner = dnd::shared_ptr<PyArrayObject>(obj, py_decref_function<PyArrayObject>);
    } else {
        Py_INCREF(base);
        bufowner = dnd::shared_ptr<PyObject>(base, py_decref_function<PyObject>);
    }
    // Create the result ndarray
    return ndarray(new strided_array_expr_node(d, PyArray_NDIM(obj), 
                    PyArray_DIMS(obj), PyArray_STRIDES(obj), PyArray_DATA(obj), DND_MOVE(bufowner)));
}

void pydnd::ndarray_init(dnd::ndarray& n, PyObject* obj)
{
    if (PyArray_Check(obj)) {
        n = from_numpy_array((PyArrayObject *)obj);
    } else {
        dtype d = pydnd::deduce_dtype_from_object(obj);
        n = ndarray(d);
    }
}

void pydnd::ndarray_init(dnd::ndarray& n, PyObject* obj, const dnd::dtype& d)
{
    if (PyArray_Check(obj)) {
        n = from_numpy_array((PyArrayObject *)obj);
        n = n.as_dtype(d);
    } else {
        n = ndarray(d);
    }
}

dnd::ndarray pydnd::ndarray_vals(const dnd::ndarray& n)
{
    return n.vals();
}
