//
// This header defines some functions to
// interoperate with numpy
//

#ifndef _DND__NUMPY_INTEROP_HPP_
#define _DND__NUMPY_INTEROP_HPP_

// Define this to 1 or 0 depending on whether numpy interop
// should be compiled in.
#define DND_NUMPY_INTEROP 1

// Only expose the things in this header when numpy interop is enabled
#if DND_NUMPY_INTEROP

// Don't use the deprecated Numpy functions
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL pydnd_ARRAY_API
// Invert the importing signal to match how numpy wants it
#ifndef NUMPY_IMPORT_ARRAY
# define NO_IMPORT_ARRAY
#endif

#include <stdint.h>
#include <sstream>

#include <dnd/dtype.hpp>
#include <dnd/ndarray.hpp>

#include "Python.h"

#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>

namespace pydnd {

inline int import_numpy()
{
#ifdef NUMPY_IMPORT_ARRAY
    import_array1(-1);
    import_umath1(-1);
#endif

    return 0;
}

/**
 * Converts a numpy dtype to a dnd::dtype.
 */
dnd::dtype dtype_from_numpy_dtype(PyArray_Descr *d);

/**
 * Gets the dtype of a numpy scalar object
 */
dnd::dtype dtype_of_numpy_scalar(PyObject* obj);

/**
 * Views a Numpy PyArrayObject as a dnd::ndarray.
 */
dnd::ndarray ndarray_from_numpy_array(PyArrayObject* obj);

} // namespace pydnd

#endif // DND_NUMPY_INTEROP

// Make a no-op import_numpy for Cython to call,
// so it doesn't need to know about DND_NUMPY_INTEROP
#if !DND_NUMPY_INTEROP
namespace pydnd {

inline int import_numpy()
{
    return 0;
}

} // namespace pydnd
#endif // !DND_NUMPY_INTEROP

#endif // _DND__NUMPY_INTEROP_HPP_