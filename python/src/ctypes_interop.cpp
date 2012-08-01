//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <dnd/dtypes/fixedstring_dtype.hpp>
#include <dnd/dtypes/pointer_dtype.hpp>

#include "ctypes_interop.hpp"
#include "dtype_functions.hpp"
#include "utility_functions.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;

ctypes_info pydnd::ctypes;


void pydnd::init_ctypes_interop()
{
    memset(&ctypes, 0, sizeof(ctypes));

    // The C _ctypes module
    ctypes._ctypes = PyImport_ImportModule("_ctypes");
    if (ctypes._ctypes == NULL) {
        throw runtime_error("Could not import module _ctypes");
    }

    // The internal type objects used by ctypes
    ctypes.PyCStructType_Type = PyObject_GetAttrString(ctypes._ctypes, "Structure");
    // _ctypes doesn't expose PyCData_Type, but we know it's the base class of PyCStructType_Type
    ctypes.PyCData_Type = (PyObject *)((PyTypeObject *)ctypes.PyCStructType_Type)->tp_base;
    ctypes.UnionType_Type = PyObject_GetAttrString(ctypes._ctypes, "Union");
    ctypes.PyCPointerType_Type = PyObject_GetAttrString(ctypes._ctypes, "_Pointer");
    ctypes.PyCArrayType_Type = PyObject_GetAttrString(ctypes._ctypes, "Array");
    ctypes.PyCSimpleType_Type = PyObject_GetAttrString(ctypes._ctypes, "_SimpleCData");
    ctypes.PyCFuncPtrType_Type = PyObject_GetAttrString(ctypes._ctypes, "CFuncPtr");

    if (PyErr_Occurred()) {
        Py_XDECREF(ctypes._ctypes);

        Py_XDECREF(ctypes.PyCData_Type);
        Py_XDECREF(ctypes.PyCStructType_Type);
        Py_XDECREF(ctypes.UnionType_Type);
        Py_XDECREF(ctypes.PyCPointerType_Type);
        Py_XDECREF(ctypes.PyCArrayType_Type);
        Py_XDECREF(ctypes.PyCSimpleType_Type);
        Py_XDECREF(ctypes.PyCFuncPtrType_Type);

        memset(&ctypes, 0, sizeof(ctypes));
        throw std::runtime_error("Error initializing ctypes C-level data for low level interop");
    }
}


calling_convention_t pydnd::get_ctypes_calling_convention(PyCFuncPtrObject* cfunc)
{
    // This is the internal StgDictObject "flags" attribute, which is
    // custom-placed in the typeobject's dict by ctypes.
    pyobject_ownref flags_obj(PyObject_GetAttrString((PyObject *)Py_TYPE(cfunc), "_flags_"));

    long flags = PyInt_AsLong(flags_obj);
    if (flags == -1 && PyErr_Occurred()) {
        throw std::runtime_error("Error getting ctypes function flags");
    }

    if (flags&0x02) { // 0x02 is FUNCFLAG_HRESULT
        throw std::runtime_error("Functions returning an HRESULT are not supported");
    }

    //if (flags&0x04) { // 0x04 is FUNCFLAG_PYTHONAPI, may need special handling
    //}

    if (flags&0x08) { // 0x08 is FUNCFLAG_USE_ERRNO
        throw std::runtime_error("Functions using errno are not yet supported");
    }

    if (flags&0x10) { // 0x10 is FUNCFLAG_USE_LASTERROR
        throw std::runtime_error("Functions using lasterror are not yet supported");
    }

    // Only on 32-bit Windows are non-CDECL calling conventions supported
#if defined(_WIN32) && !defined(_M_X64)
    if (cfunc->index) {
        throw std::runtime_error("COM functions are not supported");
    }
    if (flags&0x01) { // 0x01 is FUNCFLAG_CDECL from cpython's internal ctypes.h
        return cdecl_callconv;
    } else {
        return win32_stdcall_callconv;
    }
#else
    return cdecl_callconv;
#endif
}

void pydnd::get_ctypes_signature(PyCFuncPtrObject* cfunc, std::vector<dnd::dtype>& out_sig)
{
    // The fields restype and argtypes are not always stored at the C level,
    // so must use the higher level getattr.
    pyobject_ownref restype(PyObject_GetAttrString((PyObject *)cfunc, "restype"));
    pyobject_ownref argtypes(PyObject_GetAttrString((PyObject *)cfunc, "argtypes"));

    if (argtypes == Py_None) {
        throw std::runtime_error("The argtypes and restype of a ctypes function pointer must be specified to get its signature");
    }

    Py_ssize_t argcount = PySequence_Size(argtypes);
    if (argcount < 0) {
        throw runtime_error("The argtypes of the ctypes function pointer has the wrong type");
    }

    // Set the output size
    out_sig.resize(argcount + 1);

    // Get the return type
    if (restype == Py_None) {
        // No return type
        out_sig[0] = make_dtype<void>();
    } else {
        out_sig[0] = dtype_from_ctypes_cdatatype(restype);
    }

    // Get the argument types
    for (intptr_t i = 0; i < argcount; ++i) {
        pyobject_ownref element(PySequence_GetItem(argtypes, i));
        out_sig[i + 1] = dtype_from_ctypes_cdatatype(element);
    }
}


dnd::dtype pydnd::dtype_from_ctypes_cdatatype(PyObject *d)
{
    if (!PyObject_IsSubclass(d, ctypes.PyCData_Type)) {
        throw runtime_error("requested a dtype from a ctypes c data type, but the given object has the wrong type");
    }

    // If the ctypes type has a _dynd_type_ property, that should be
    // a pydnd dtype instance corresponding to the type. This is how
    // the complex type is supported, for example.
    PyObject *dynd_type_obj = PyObject_GetAttrString(d, "_dynd_type_");
    if (dynd_type_obj == NULL) {
        PyErr_Clear();
    } else {
        pyobject_ownref dynd_type(dynd_type_obj);
        return make_dtype_from_object(dynd_type);
    }

    // The simple C data types
    if (PyObject_IsSubclass(d, ctypes.PyCSimpleType_Type)) {
        char *proto_str = NULL;
        Py_ssize_t proto_len = 0;
        pyobject_ownref proto(PyObject_GetAttrString(d, "_type_"));
        if (PyString_AsStringAndSize(proto, &proto_str, &proto_len) < 0 ||
                            proto_len != 1) {
            throw std::runtime_error("invalid ctypes type");
        }

        switch (proto_str[0]) {
            case 'b':
                return make_dtype<int8_t>();
            case 'B':
                return make_dtype<uint8_t>();
            case 'c':
                return make_fixedstring_dtype(string_encoding_ascii, 1);
            case 'd':
                return make_dtype<double>();
            case 'f':
                return make_dtype<float>();
            case 'h':
                return make_dtype<int16_t>();
            case 'H':
                return make_dtype<uint16_t>();
            case 'i':
                return make_dtype<int32_t>();
            case 'I':
                return make_dtype<uint32_t>();
            case 'l':
                return make_dtype<long>();
            case 'L':
                return make_dtype<unsigned long>();
            case 'q':
                return make_dtype<int64_t>();
            case 'Q':
                return make_dtype<uint64_t>();
            default: {
                stringstream ss;
                ss << "The ctypes type code '" << proto_str[0] << "' cannot be converted to a dnd::dtype";
                throw runtime_error(ss.str());
            }
        }
    } else if (PyObject_IsSubclass(d, ctypes.PyCPointerType_Type)) {
        // Translate into a blockref pointer dtype
        pyobject_ownref target_dtype_obj(PyObject_GetAttrString(d, "_type_"));
        dtype target_dtype = dtype_from_ctypes_cdatatype(target_dtype_obj);
        return make_pointer_dtype(target_dtype);
    }

    throw runtime_error("Ctypes type object is not supported by dnd::dtype");
}
