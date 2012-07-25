//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__UTILITY_FUNCTIONS_HPP_
#define _DND__UTILITY_FUNCTIONS_HPP_

#include <stdint.h>
#include <sstream>
#include <stdexcept>

#include <dnd/dtype.hpp>

#include "Python.h"

namespace pydnd {

/**
 * A container class for managing the local lifetime of
 * PyObject *.
 *
 * Throws an exception if the object passed into the constructor
 * is NULL.
 */
class pyobject_ownref {
    PyObject *m_obj;

    // Non-copyable
    pyobject_ownref(const pyobject_ownref&);
    pyobject_ownref& operator=(const pyobject_ownref&);
public:
    explicit pyobject_ownref(PyObject* obj)
        : m_obj(obj)
    {
        if (obj == NULL) {
            throw std::runtime_error("propagating a Python exception...need a mechanism to do that through Cython with a C++ exception");
        }
    }

    ~pyobject_ownref()
    {
        Py_XDECREF(m_obj);
    }

    PyObject *get()
    {
        return m_obj;
    }

    // Returns a borrowed reference
    operator PyObject *()
    {
        return m_obj;
    }

    /**
     * Returns the reference owned by this object,
     * use it like "return obj.release()". After the
     * call, this object contains NULL.
     */
    PyObject *release()
    {
        PyObject *result = m_obj;
        m_obj = NULL;
        return result;
    }
};

intptr_t pyobject_as_index(PyObject *index);

PyObject* intptr_array_as_tuple(int size, const intptr_t *array);

/**
 * Parses the axis argument, which may be either a single index
 * or a tuple of indices. They are converted into a boolean array
 * which is set to true whereever a reduction axis is provided.
 */
void pyarg_axis_argument(PyObject *axis, int ndim, dnd::dnd_bool *reduce_axes);

/**
 * Matches the input object against one of several
 * strings, returning the corresponding integer.
 */
int pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                const char *string0, int value0);

/**
 * Matches the input object against one of several
 * strings, returning the corresponding integer.
 */
int pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                const char *string0, int value0,
                const char *string1, int value1);

bool pyarg_bool(PyObject *obj, const char *argname, bool default_value);

} // namespace pydnd

#endif // _DND__UTILITY_FUNCTIONS_HPP_
