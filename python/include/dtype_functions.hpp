//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some wrapping functions to
// access various dtype parameters
//

#ifndef _DND__DTYPE_FUNCTIONS_HPP_
#define _DND__DTYPE_FUNCTIONS_HPP_

#include <stdint.h>
#include <sstream>

#include <dnd/dtype.hpp>
#include <dnd/string_encodings.hpp>

#include "Python.h"

namespace pydnd {

/**
 * This is the typeobject and struct of w_dtype from Cython.
 */
extern PyTypeObject *WDType_Type;
inline bool WDType_CheckExact(PyObject *obj) {
    return Py_TYPE(obj) == WDType_Type;
}
inline bool WDType_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, WDType_Type);
}
struct WDType {
  PyObject_HEAD;
  // This is dtype_placement_wrapper in Cython-land
  dnd::dtype v;
};
void init_w_dtype_typeobject(PyObject *type);

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
 * Produces a dnd::dtype corresponding to the object's type. This
 * is to determine the dtype of an object that contains a value
 * or values.
 */
dnd::dtype deduce_dtype_from_object(PyObject* obj);

/**
 * Converts a Python type, Numpy dtype, or string
 * into a dtype. This raises an error if given an object
 * which contains values, it is for dtype-like things only.
 */
dnd::dtype make_dtype_from_object(PyObject* obj);

/**
 * Creates a fixed-sized string dtype.
 */
dnd::dtype dnd_make_fixedstring_dtype(PyObject *encoding_obj, intptr_t size);

} // namespace pydnd

#endif // _DND__DTYPE_FUNCTIONS_HPP_
