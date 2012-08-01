//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some functions to
// interoperate with ctypes
//

#ifndef _DND__CTYPES_INTEROP_HPP_
#define _DND__CTYPES_INTEROP_HPP_

#include <Python.h>

#include <dnd/dtype.hpp>
#include <dnd/ndarray.hpp>
#include <dnd/codegen/calling_conventions.hpp>

namespace pydnd {

/**
 * Struct with data about the _ctypes module.
 */
struct ctypes_info {
    // The _ctypes module (for C-implementation details)
    PyObject *_ctypes;
    // These match the corresponding names within _ctypes.c
    PyObject *PyCData_Type;
    PyObject *PyCStructType_Type;
    PyObject *UnionType_Type;
    PyObject *PyCPointerType_Type;
    PyObject *PyCArrayType_Type;
    PyObject *PyCSimpleType_Type;
    PyObject *PyCFuncPtrType_Type;
};

extern ctypes_info ctypes;

/**
 * Should be called at module initialization, this
 * stores some internal information about the ctypes
 * classes for later.
 */
void init_ctypes_interop();

/**
 * Constructs a dtype from a ctypes type object, such
 * as ctypes.c_int, ctypes.c_float, etc.
 */
dnd::dtype dtype_from_ctypes_cdatatype(PyObject *d);

//////////////////////////////////////////////////////////
// The following emulates a lot of the internal ctypes.h
// API, so we can directly access ctypes data quickly.
//
// TODO: Try this across many versions of Python,
//       tweak it for variations.

union ctypes_value {
    char c[16];
    short s;
    int i;
    long l;
    float f;
    double d;
#ifdef HAVE_LONG_LONG
    PY_LONG_LONG ll;
#endif
    long double D;
};

struct CDataObject {
    PyObject_HEAD
    char *b_ptr;                /* pointer to memory block */
    int  b_needsfree;           /* need _we_ free the memory? */
    CDataObject *b_base;        /* pointer to base object or NULL */
    Py_ssize_t b_size;          /* size of memory block in bytes */
    Py_ssize_t b_length;        /* number of references we need */
    Py_ssize_t b_index;         /* index of this object into base's b_object list */
    PyObject *b_objects;        /* dictionary of references we need to keep, or Py_None */
    ctypes_value b_value;
};

struct PyCFuncPtrObject {
    // First part identical to CDataObject
    PyObject_HEAD
    char *b_ptr;                /* pointer to memory block */
    int  b_needsfree;           /* need _we_ free the memory? */
    CDataObject *b_base;        /* pointer to base object or NULL */
    Py_ssize_t b_size;          /* size of memory block in bytes */
    Py_ssize_t b_length;        /* number of references we need */
    Py_ssize_t b_index;         /* index of this object into base's b_object list */
    PyObject *b_objects;        /* list of references we need to keep */
    ctypes_value b_value;
    /* end of CDataObject, additional fields follow */

    void *thunk;
    PyObject *callable;

    /* These two fields will override the ones in the type's stgdict if
       they are set */
    PyObject *converters;
    PyObject *argtypes;
    PyObject *restype;
    PyObject *checker;
    PyObject *errcheck;
#ifdef _WIN32
    int index;
    void *iid;
#endif
    PyObject *paramflags;
};

struct ctypes_ffi_type
{
  size_t size;
  unsigned short alignment;
  unsigned short type;
  ctypes_ffi_type **elements;
};

// These functions correspond to functions or macros in
// CPython's internal "ctypes.h"
inline bool CDataObject_CheckExact(PyObject *v) {
    return v->ob_type == (PyTypeObject *)ctypes.PyCData_Type;
}
inline bool CDataObject_Check(PyObject *v) {
    return PyObject_TypeCheck(v, (PyTypeObject *)ctypes.PyCData_Type);
}

/**
 * Gets the calling convention of the ctypes function pointer object.
 */
dnd::calling_convention_t get_ctypes_calling_convention(PyCFuncPtrObject* cfunc);

/**
 * Gets the signature of the ctypes function pointer object. out_sig[0] is
 * the return type, and the rest are the argument types.
 */
void get_ctypes_signature(PyCFuncPtrObject* cfunc, std::vector<dnd::dtype>& out_sig);

} // namespace pydnd

#endif // _DND__CTYPES_INTEROP_HPP_
