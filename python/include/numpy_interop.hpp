//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some functions to
// interoperate with numpy
//

#ifndef _DND__NUMPY_INTEROP_HPP_
#define _DND__NUMPY_INTEROP_HPP_

#include <Python.h>

// Define this to 1 or 0 depending on whether numpy interop
// should be compiled in.
#define DND_NUMPY_INTEROP 1

// Only expose the things in this header when numpy interop is enabled
#if DND_NUMPY_INTEROP

#include <numpy/numpyconfig.h>

// Don't use the deprecated Numpy functions
#ifdef NPY_1_7_API_VERSION
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#else
# define NPY_ARRAY_NOTSWAPPED NPY_NOTSWAPPED
# define NPY_ARRAY_ALIGNED NPY_ALIGNED
# define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
# define NPY_ARRAY_UPDATEIFCOPY NPY_UPDATEIFCOPY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL pydnd_ARRAY_API
// Invert the importing signal to match how numpy wants it
#ifndef NUMPY_IMPORT_ARRAY
# define NO_IMPORT_ARRAY
#endif

#include <stdint.h>
#include <sstream>

#include <dnd/dtype.hpp>
#include <dnd/ndarray.hpp>

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
 * Converts a pytypeobject for a numpy scalar
 * into a dnd::dtype.
 *
 * Returns 0 on success, -1 if it didn't match.
 */
int dtype_from_numpy_scalar_typeobject(PyTypeObject* obj, dnd::dtype& out_d);

/**
 * Gets the dtype of a numpy scalar object
 */
dnd::dtype dtype_of_numpy_scalar(PyObject* obj);

/**
 * Views a Numpy PyArrayObject as a dnd::ndarray.
 */
dnd::ndarray ndarray_from_numpy_array(PyArrayObject* obj);

/**
 * Creates a dnd::ndarray from a numpy scalar.
 */
dnd::ndarray ndarray_from_numpy_scalar(PyObject* obj);

/**
 * Returns the numpy kind ('i', 'f', etc) of the array.
 */
char numpy_kindchar_of(const dnd::dtype& d);

/**
 * Produces a PyCapsule (or PyCObject as appropriate) which
 * contains a __array_struct__ interface object.
 */
PyObject* ndarray_as_numpy_struct_capsule(const dnd::ndarray& n);

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

// If we're not building against Numpy, define our
// own version of this struct to use.
typedef struct {
    int two;              /*
                           * contains the integer 2 as a sanity
                           * check
                           */

    int nd;               /* number of dimensions */

    char typekind;        /*
                           * kind in array --- character code of
                           * typestr
                           */

    int element_size;         /* size of each element */

    int flags;            /*
                           * how should be data interpreted. Valid
                           * flags are CONTIGUOUS (1), F_CONTIGUOUS (2),
                           * ALIGNED (0x100), NOTSWAPPED (0x200), and
                           * WRITEABLE (0x400).  ARR_HAS_DESCR (0x800)
                           * states that arrdescr field is present in
                           * structure
                           */

    npy_intp *shape;       /*
                            * A length-nd array of shape
                            * information
                            */

    npy_intp *strides;    /* A length-nd array of stride information */

    void *data;           /* A pointer to the first element of the array */

    PyObject *descr;      /*
                           * A list of fields or NULL (ignored if flags
                           * does not have ARR_HAS_DESCR flag set)
                           */
} PyArrayInterface;

} // namespace pydnd
#endif // !DND_NUMPY_INTEROP

#endif // _DND__NUMPY_INTEROP_HPP_
