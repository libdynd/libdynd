#include "ndarray_functions.hpp"
#include "dtype_functions.hpp"
#include "numpy_interop.hpp"

using namespace std;
using namespace dnd;

void pydnd::ndarray_init(dnd::ndarray& n, PyObject* obj)
{
#if DND_NUMPY_INTEROP
    if (PyArray_Check(obj)) {
        n = ndarray_from_numpy_array((PyArrayObject *)obj);
    }
#endif // DND_NUMPY_INTEROP

    dtype d = pydnd::deduce_dtype_from_object(obj);
    n = ndarray(d);
}

void pydnd::ndarray_init(dnd::ndarray& n, PyObject* obj, const dnd::dtype& d)
{
#if DND_NUMPY_INTEROP
    if (PyArray_Check(obj)) {
        n = ndarray_from_numpy_array((PyArrayObject *)obj);
        n = n.as_dtype(d);
        return;
    }
#endif // DND_NUMPY_INTEROP

    n = ndarray(d);
}

dnd::ndarray pydnd::ndarray_vals(const dnd::ndarray& n)
{
    return n.vals();
}
