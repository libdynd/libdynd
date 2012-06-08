#include "dtype_functions.hpp"
#include "ndarray_functions.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;

dtype pydnd::dtype_from_numpy_dtype(PyArray_Descr *d)
{
    if (!PyArray_ISNBO(d->byteorder)) {
        throw runtime_error("non-native byte order isn't supported yet by pydnd");
    }

    switch (d->type_num) {
    case NPY_BOOL:
        return make_dtype<dnd_bool>();
    case NPY_BYTE:
        return make_dtype<npy_byte>();
    case NPY_UBYTE:
        return make_dtype<npy_ubyte>();
    case NPY_SHORT:
        return make_dtype<npy_short>();
    case NPY_USHORT:
        return make_dtype<npy_ushort>();
    case NPY_INT:
        return make_dtype<npy_int>();
    case NPY_UINT:
        return make_dtype<npy_uint>();
    case NPY_LONG:
        return make_dtype<npy_long>();
    case NPY_ULONG:
        return make_dtype<npy_ulong>();
    case NPY_LONGLONG:
        return make_dtype<npy_longlong>();
    case NPY_ULONGLONG:
        return make_dtype<npy_ulonglong>();
    case NPY_FLOAT:
        return make_dtype<float>();
    case NPY_DOUBLE:
        return make_dtype<double>();
    case NPY_CFLOAT:
        return make_dtype<complex<float> >();
    case NPY_CDOUBLE:
        return make_dtype<complex<double> >();
    }

    stringstream ss;
    ss << "unsupported Numpy dtype with type id " << d->type_num;
    throw runtime_error(ss.str());
}

dtype pydnd::deduce_dtype_from_object(PyObject* obj)
{
    if (PyArray_Check(obj)) {
        PyArray_Descr *d = PyArray_DESCR((PyArrayObject *)obj);
        return dtype_from_numpy_dtype(d);
    } else if (PyBool_Check(obj)) {
        return make_dtype<dnd_bool>();
    } else if (PyLong_Check(obj) || PyInt_Check(obj)) {
        return make_dtype<int32_t>();
    } else if (PyFloat_Check(obj)) {
        return make_dtype<double>();
    } else if (PyComplex_Check(obj)) {
        return make_dtype<complex<double> >();
    } else if (PyArray_IsScalar(obj, Generic)) {
        if (PyArray_IsScalar(obj, Bool)) {
            return make_dtype<dnd_bool>();
        } else if (PyArray_IsScalar(obj, Int8)) {
            return make_dtype<int8_t>();
        } else if (PyArray_IsScalar(obj, Int16)) {
            return make_dtype<int16_t>();
        } else if (PyArray_IsScalar(obj, Int32)) {
            return make_dtype<int32_t>();
        } else if (PyArray_IsScalar(obj, Int64)) {
            return make_dtype<int64_t>();
        } else if (PyArray_IsScalar(obj, UInt8)) {
            return make_dtype<uint8_t>();
        } else if (PyArray_IsScalar(obj, UInt16)) {
            return make_dtype<uint16_t>();
        } else if (PyArray_IsScalar(obj, UInt32)) {
            return make_dtype<uint32_t>();
        } else if (PyArray_IsScalar(obj, UInt64)) {
            return make_dtype<uint64_t>();
        } else if (PyArray_IsScalar(obj, Float32)) {
            return make_dtype<float>();
        } else if (PyArray_IsScalar(obj, Float64)) {
            return make_dtype<double>();
        } else if (PyArray_IsScalar(obj, Complex64)) {
            return make_dtype<complex<float> >();
        } else if (PyArray_IsScalar(obj, Complex128)) {
            return make_dtype<complex<double> >();
        }

        throw std::runtime_error("could not deduce pydnd dtype from numpy scalar object");
    }

    throw std::runtime_error("could not deduce pydnd dtype from python object");
}
