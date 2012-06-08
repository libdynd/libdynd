//
// This header defines some wrapping functions to
// access various dtype parameters
//

#ifndef _DND__DTYPE_FUNCTIONS_HPP_
#define _DND__DTYPE_FUNCTIONS_HPP_

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

#include "Python.h"
#include <numpy/ndarrayobject.h>

namespace pydnd {

inline std::string dtype_str(const dnd::dtype& d)
{
    std::stringstream ss;
    ss << d;
    return ss.str();
}

inline std::string dtype_repr(const dnd::dtype& d)
{
    std::stringstream ss;
    if (d.type_id() < dnd::builtin_type_id_count &&
                    d.type_id() != dnd::complex_float32_type_id &&
                    d.type_id() != dnd::complex_float64_type_id) {
        ss << "pydnd." << d;
    } else {
        ss << "pydnd.dtype('" << d << "')";
    }
    return ss.str();
}

/**
 * Converts a numpy dtype to a dnd::dtype.
 */
dnd::dtype dtype_from_numpy_dtype(PyArray_Descr *d);

/**
 * Produces a dnd::dtype corresponding to the object's type
 */
dnd::dtype deduce_dtype_from_object(PyObject* obj);

} // namespace pydnd

#endif // _DND__DTYPE_FUNCTIONS_HPP_