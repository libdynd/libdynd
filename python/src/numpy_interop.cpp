//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "numpy_interop.hpp"

#if DND_NUMPY_INTEROP

#include <dnd/dtypes/byteswap_dtype.hpp>
#include <dnd/dtypes/view_dtype.hpp>
#include <dnd/dtypes/dtype_alignment.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>

#include "dtype_functions.hpp"
#include "ndarray_functions.hpp"

#include <numpy/arrayscalars.h>

using namespace std;
using namespace dnd;
using namespace pydnd;

dtype pydnd::dtype_from_numpy_dtype(PyArray_Descr *d)
{
    dtype dt;

    switch (d->type_num) {
    case NPY_BOOL:
        dt = make_dtype<dnd_bool>();
        break;
    case NPY_BYTE:
        dt = make_dtype<npy_byte>();
        break;
    case NPY_UBYTE:
        dt = make_dtype<npy_ubyte>();
        break;
    case NPY_SHORT:
        dt = make_dtype<npy_short>();
        break;
    case NPY_USHORT:
        dt = make_dtype<npy_ushort>();
        break;
    case NPY_INT:
        dt = make_dtype<npy_int>();
        break;
    case NPY_UINT:
        dt = make_dtype<npy_uint>();
        break;
    case NPY_LONG:
        dt = make_dtype<npy_long>();
        break;
    case NPY_ULONG:
        dt = make_dtype<npy_ulong>();
        break;
    case NPY_LONGLONG:
        dt = make_dtype<npy_longlong>();
        break;
    case NPY_ULONGLONG:
        dt = make_dtype<npy_ulonglong>();
        break;
    case NPY_FLOAT:
        dt = make_dtype<float>();
        break;
    case NPY_DOUBLE:
        dt = make_dtype<double>();
        break;
    case NPY_CFLOAT:
        dt = make_dtype<complex<float> >();
        break;
    case NPY_CDOUBLE:
        dt = make_dtype<complex<double> >();
        break;
    case NPY_STRING:
        dt = make_fixedstring_dtype(string_encoding_ascii, d->elsize);
        break;
    case NPY_UNICODE:
        dt = make_fixedstring_dtype(string_encoding_utf_32, d->elsize / 4);
        break;
    default: {
        stringstream ss;
        ss << "unsupported Numpy dtype with type id " << d->type_num;
        throw runtime_error(ss.str());
        }
    }

    if (!PyArray_ISNBO(d->byteorder)) {
        dt = make_byteswap_dtype(dt);
    }

    return dt;
}

int pydnd::dtype_from_numpy_scalar_typeobject(PyTypeObject* obj, dnd::dtype& out_d)
{
    if (obj == &PyBoolArrType_Type) {
        out_d = make_dtype<dnd_bool>();
    } else if (obj == &PyByteArrType_Type) {
        out_d = make_dtype<npy_byte>();
    } else if (obj == &PyUByteArrType_Type) {
        out_d = make_dtype<npy_ubyte>();
    } else if (obj == &PyShortArrType_Type) {
        out_d = make_dtype<npy_short>();
    } else if (obj == &PyUShortArrType_Type) {
        out_d = make_dtype<npy_ushort>();
    } else if (obj == &PyIntArrType_Type) {
        out_d = make_dtype<npy_int>();
    } else if (obj == &PyUIntArrType_Type) {
        out_d = make_dtype<npy_uint>();
    } else if (obj == &PyLongArrType_Type) {
        out_d = make_dtype<npy_long>();
    } else if (obj == &PyULongArrType_Type) {
        out_d = make_dtype<npy_ulong>();
    } else if (obj == &PyLongLongArrType_Type) {
        out_d = make_dtype<npy_longlong>();
    } else if (obj == &PyULongLongArrType_Type) {
        out_d = make_dtype<npy_ulonglong>();
    } else if (obj == &PyFloatArrType_Type) {
        out_d = make_dtype<npy_float>();
    } else if (obj == &PyDoubleArrType_Type) {
        out_d = make_dtype<npy_double>();
    } else if (obj == &PyCFloatArrType_Type) {
        out_d = make_dtype<complex<float> >();
    } else if (obj == &PyCDoubleArrType_Type) {
        out_d = make_dtype<complex<double> >();
    } else {
        return -1;
    }

    return 0;
}

dtype pydnd::dtype_of_numpy_scalar(PyObject* obj)
{
    if (PyArray_IsScalar(obj, Bool)) {
        return make_dtype<dnd_bool>();
    } else if (PyArray_IsScalar(obj, Byte)) {
        return make_dtype<npy_byte>();
    } else if (PyArray_IsScalar(obj, UByte)) {
        return make_dtype<npy_ubyte>();
    } else if (PyArray_IsScalar(obj, Short)) {
        return make_dtype<npy_short>();
    } else if (PyArray_IsScalar(obj, UShort)) {
        return make_dtype<npy_ushort>();
    } else if (PyArray_IsScalar(obj, Int)) {
        return make_dtype<npy_int>();
    } else if (PyArray_IsScalar(obj, UInt)) {
        return make_dtype<npy_uint>();
    } else if (PyArray_IsScalar(obj, Long)) {
        return make_dtype<npy_long>();
    } else if (PyArray_IsScalar(obj, ULong)) {
        return make_dtype<npy_ulong>();
    } else if (PyArray_IsScalar(obj, LongLong)) {
        return make_dtype<npy_longlong>();
    } else if (PyArray_IsScalar(obj, ULongLong)) {
        return make_dtype<npy_ulonglong>();
    } else if (PyArray_IsScalar(obj, Float)) {
        return make_dtype<float>();
    } else if (PyArray_IsScalar(obj, Double)) {
        return make_dtype<double>();
    } else if (PyArray_IsScalar(obj, CFloat)) {
        return make_dtype<complex<float> >();
    } else if (PyArray_IsScalar(obj, CDouble)) {
        return make_dtype<complex<double> >();
    }

    throw std::runtime_error("could not deduce a pydnd dtype from the numpy scalar object");
}

template<class T>
static void py_decref_function(T* obj)
{
    Py_DECREF(obj);
}

ndarray pydnd::ndarray_from_numpy_array(PyArrayObject* obj)
{
    // Get the dtype of the array
    dtype d = pydnd::dtype_from_numpy_dtype(PyArray_DESCR(obj));

    // If the array's data isn't aligned properly, apply better alignment
    if (((uintptr_t)PyArray_DATA(obj)&(d.alignment()-1)) != 0) {
        d = make_unaligned_dtype(d);
    } else {
        int ndim = PyArray_NDIM(obj);
        intptr_t *strides = PyArray_STRIDES(obj);
        for (int idim = 0; idim < ndim; ++idim) {
            if (((uintptr_t)strides[idim]&(d.alignment()-1)) != 0) {
                d = make_unaligned_dtype(d);
                break;
            }
        }
    }

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
                    PyArray_DIMS(obj), PyArray_STRIDES(obj), PyArray_BYTES(obj), DND_MOVE(bufowner)));
}

dnd::ndarray pydnd::ndarray_from_numpy_scalar(PyObject* obj)
{
    if (PyArray_IsScalar(obj, Bool)) {
        return ndarray((bool)((PyBoolScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Byte)) {
        return ndarray(((PyByteScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, UByte)) {
        return ndarray(((PyUByteScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Short)) {
        return ndarray(((PyShortScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, UShort)) {
        return ndarray(((PyUShortScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Int)) {
        return ndarray(((PyIntScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, UInt)) {
        return ndarray(((PyUIntScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Long)) {
        return ndarray(((PyLongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, ULong)) {
        return ndarray(((PyULongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, LongLong)) {
        return ndarray(((PyLongLongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, ULongLong)) {
        return ndarray(((PyULongLongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Float)) {
        return ndarray(((PyFloatScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Double)) {
        return ndarray(((PyDoubleScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, CFloat)) {
        npy_cfloat& val = ((PyCFloatScalarObject *)obj)->obval;
        return ndarray(complex<float>(val.real, val.imag));
    } else if (PyArray_IsScalar(obj, CDouble)) {
        npy_cdouble& val = ((PyCDoubleScalarObject *)obj)->obval;
        return ndarray(complex<double>(val.real, val.imag));
    }

    throw std::runtime_error("could not create a dnd::ndarray from the numpy scalar object");
}

char pydnd::numpy_kindchar_of(const dnd::dtype& d)
{
    switch (d.kind()) {
    case bool_kind:
        return 'b';
    case int_kind:
        return 'i';
    case uint_kind:
        return 'u';
    case real_kind:
        return 'f';
    case complex_kind:
        return 'c';
    case string_kind:
        switch (d.string_encoding()) {
        case string_encoding_ascii:
            return 'S';
        case string_encoding_utf_32:
            return 'U';
        default:
            break;
        }
    default: {
        stringstream ss;
        ss << "dnd::dtype \"" << d << "\" does not have an equivalent numpy kind";
        throw runtime_error(ss.str());
        }
    }
}

#endif // DND_NUMPY_INTEROP

// The function ndarray_as_numpy_struct_capsule is exposed even without building against numpy
static void free_array_interface(void *ptr, void *extra_ptr)
{
    PyArrayInterface* inter = (PyArrayInterface *)ptr;
    dnd::shared_ptr<void> *extra = (dnd::shared_ptr<void> *)extra_ptr;
    delete[] inter->strides;
    delete inter;
    delete extra;
}

PyObject* pydnd::ndarray_as_numpy_struct_capsule(const dnd::ndarray& n)
{
    if (n.get_expr_tree()->get_node_type() != strided_array_node_type) {
        throw runtime_error("cannot convert a dnd::ndarray that isn't a strided array into a numpy array");
    }

    dtype dt = n.get_dtype();
    dtype value_dt = dt.value_dtype();

    bool byteswapped = false;
    if (dt.type_id() == byteswap_type_id) {
        dt = dt.operand_dtype();
        byteswapped = true;
    }

    bool aligned = true;
    if (dt.type_id() == view_type_id) {
        dtype sdt = dt.operand_dtype();
        if (sdt.type_id() == bytes_type_id) {
            dt = dt.value_dtype();
            aligned = false;
        }
    }

    PyArrayInterface inter;
    memset(&inter, 0, sizeof(inter));

    inter.two = 2;
    inter.nd = n.get_ndim();
    inter.typekind = numpy_kindchar_of(value_dt);
    // Numpy treats 'U' as number of 4-byte characters, not number of bytes
    inter.itemsize = (int)(inter.typekind != 'U' ? n.get_dtype().element_size() : n.get_dtype().element_size() / 4);
    // TODO: When read-write access control is added, this must be modified
    inter.flags = (byteswapped ? 0 : NPY_ARRAY_NOTSWAPPED) | (aligned ? NPY_ARRAY_ALIGNED : 0) | NPY_ARRAY_WRITEABLE;
    inter.data = n.get_originptr();
    inter.strides = new intptr_t[2 * n.get_ndim()];
    inter.shape = inter.strides + n.get_ndim();

    memcpy(inter.strides, n.get_strides(), n.get_ndim() * sizeof(intptr_t));
    memcpy(inter.shape, n.get_shape(), n.get_ndim() * sizeof(intptr_t));

    // TODO: Check for Python 3, use PyCapsule there
    return PyCObject_FromVoidPtrAndDesc(new PyArrayInterface(inter), new dnd::shared_ptr<void>(n.get_buffer_owner()), free_array_interface);
}
