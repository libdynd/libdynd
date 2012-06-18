//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "unary_gfunc.hpp"
#include "ndarray_functions.hpp"
#include "utility_functions.hpp"
#include "ctypes_interop.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;

namespace {

    template<typename S, typename T>
    struct unary_kernels {
        typedef S (*cdecl_func_ptr_t)(T);

        static void cdecl_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                                    intptr_t count, const AuxDataBase *auxdata)
        {
            cdecl_func_ptr_t kfunc = get_auxiliary_data<cdecl_func_ptr_t>(auxdata);

            for (intptr_t i = 0; i < count; ++i) {
                *(S *)dst = kfunc(*(const T *)src);
                dst += dst_stride;
                src += src_stride;
            }
        }

#ifdef _WIN32
        typedef S (__stdcall *win32_stdcall_func_ptr_t)(T);

        static void win32_stdcall_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                                    intptr_t count, const AuxDataBase *auxdata)
        {
            win32_stdcall_func_ptr_t kfunc = get_auxiliary_data<win32_stdcall_func_ptr_t>(auxdata);

            for (intptr_t i = 0; i < count; ++i) {
                *(S *)dst = kfunc(*(const T *)src);
                dst += dst_stride;
                src += src_stride;
            }
        }
#endif
    }; // struct unary_kernels

} // anonymous namespace

static void create_unary_gfunc_kernel_from_ctypes(PyCFuncPtrObject *cfunc, unary_gfunc_kernel& out_kernel)
{
    // The fields restype and argtypes are not always stored at the C level,
    // so must use Python level getattr.
    pyobject_ownref restype(PyObject_GetAttrString((PyObject *)cfunc, "restype"));
    pyobject_ownref argtypes(PyObject_GetAttrString((PyObject *)cfunc, "argtypes"));

    if (argtypes == Py_None) {
        throw std::runtime_error("To construct a gfunc kernel from a ctypes function, its prototype must be fully specified");
    }
    if (restype == Py_None) {
        // TODO: May want support for this later, e.g. for print or other output functions
        throw std::runtime_error("Cannot construct a gfunc kernel from a ctypes function which returns void");
    }

    Py_ssize_t argcount = PySequence_Size(argtypes);

    if (argcount != 1) {
        std::stringstream ss;
        ss << "Only unary gfunc kernels are currently supported, provided gfunc has " << argcount;
        throw std::runtime_error(ss.str());
    }

    out_kernel.m_out = dtype_from_ctypes_cdatatype(restype);
    for (int i = 0; i < argcount; ++i) {
        pyobject_ownref item(PySequence_GetItem(argtypes, i));
        out_kernel.m_params[i] = dtype_from_ctypes_cdatatype(item);
    }

    if (out_kernel.m_out != out_kernel.m_params[0]) {
        throw std::runtime_error("Only gfunc kernels with matching input and output are currently supported");
    }

    ctypes_calling_convention cc = get_ctypes_calling_convention(cfunc);
    cout << "calling convention is " << cc << endl;
#ifdef _WIN32
    if (cc == win32_stdcall_callconv) {
        switch (out_kernel.m_params[0].type_id()) {
            case int8_type_id:
                out_kernel.m_kernel.kernel = &unary_kernels<int8_t, int8_t>::win32_stdcall_kernel;
                break;
            case int16_type_id:
                out_kernel.m_kernel.kernel = &unary_kernels<int16_t, int16_t>::win32_stdcall_kernel;
                break;
            case int32_type_id:
                out_kernel.m_kernel.kernel = &unary_kernels<int32_t, int32_t>::win32_stdcall_kernel;
                break;
            case int64_type_id:
                out_kernel.m_kernel.kernel = &unary_kernels<int64_t, int64_t>::win32_stdcall_kernel;
                break;
            case uint8_type_id:
                out_kernel.m_kernel.kernel = &unary_kernels<uint8_t, int8_t>::win32_stdcall_kernel;
                break;
            case uint16_type_id:
                out_kernel.m_kernel.kernel = &unary_kernels<uint16_t, uint16_t>::win32_stdcall_kernel;
                break;
            case uint32_type_id:
                out_kernel.m_kernel.kernel = &unary_kernels<uint32_t, uint32_t>::win32_stdcall_kernel;
                break;
            case uint64_type_id:
                out_kernel.m_kernel.kernel = &unary_kernels<uint64_t, uint64_t>::win32_stdcall_kernel;
                break;
            case float32_type_id:
                out_kernel.m_kernel.kernel = &unary_kernels<float, float>::win32_stdcall_kernel;
                break;
            case float64_type_id:
                out_kernel.m_kernel.kernel = &unary_kernels<double, double>::win32_stdcall_kernel;
                break;
            default:
                throw std::runtime_error("Couldn't construct a kernel for the ctypes function prototype");
        }
    }
#endif

    switch (out_kernel.m_params[0].type_id()) {
        case int8_type_id:
            out_kernel.m_kernel.kernel = &unary_kernels<int8_t, int8_t>::cdecl_kernel;
            break;
        case int16_type_id:
            out_kernel.m_kernel.kernel = &unary_kernels<int16_t, int16_t>::cdecl_kernel;
            break;
        case int32_type_id:
            out_kernel.m_kernel.kernel = &unary_kernels<int32_t, int32_t>::cdecl_kernel;
            break;
        case int64_type_id:
            out_kernel.m_kernel.kernel = &unary_kernels<int64_t, int64_t>::cdecl_kernel;
            break;
        case uint8_type_id:
            out_kernel.m_kernel.kernel = &unary_kernels<uint8_t, int8_t>::cdecl_kernel;
            break;
        case uint16_type_id:
            out_kernel.m_kernel.kernel = &unary_kernels<uint16_t, uint16_t>::cdecl_kernel;
            break;
        case uint32_type_id:
            out_kernel.m_kernel.kernel = &unary_kernels<uint32_t, uint32_t>::cdecl_kernel;
            break;
        case uint64_type_id:
            out_kernel.m_kernel.kernel = &unary_kernels<uint64_t, uint64_t>::cdecl_kernel;
            break;
        case float32_type_id:
            out_kernel.m_kernel.kernel = &unary_kernels<float, float>::cdecl_kernel;
            break;
        case float64_type_id:
            out_kernel.m_kernel.kernel = &unary_kernels<double, double>::cdecl_kernel;
            break;
        default:
            throw std::runtime_error("Couldn't construct a kernel for the ctypes function prototype");
    }

    // Put the function pointer in some auxiliary data
    make_auxiliary_data<void *>(out_kernel.m_kernel.auxdata, *(void **)cfunc->b_ptr);
}

void pydnd::unary_gfunc::add_kernel(PyObject *kernel)
{
    if (PyObject_IsSubclass((PyObject *)Py_TYPE(kernel), ctypes.PyCFuncPtrType_Type)) {
        unary_gfunc_kernel ugk;
        create_unary_gfunc_kernel_from_ctypes((PyCFuncPtrObject *)kernel, ugk);
        m_kernels.push_back(unary_gfunc_kernel());
        ugk.swap(m_kernels.back());
        return;
    }

    throw std::runtime_error("Object could not be used as a gfunc kernel");
}

PyObject *pydnd::unary_gfunc::call(PyObject *args, PyObject *kwargs)
{
    if (PySequence_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Unary gfuncs only take one argument");
        return NULL;
    }

    pyobject_ownref arg0_obj(PySequence_GetItem(args, 0));
    ndarray arg0;
    ndarray_init_from_pyobject(arg0, arg0_obj);

    type_id_t tid = arg0.get_dtype().type_id();
    for (int i = 0; i < m_kernels.size(); ++i) {
        if (tid == m_kernels[i].m_params[0].type_id()) {
        }
    }

    std::stringstream ss;
    ss << "Could not find a gfunc kernel matching dtype " << arg0.get_dtype();
    throw std::runtime_error(ss.str());

}

std::string pydnd::unary_gfunc::debug_dump() const
{
    std::stringstream o;
    o << "------ unary_gfunc\n";
    o << "name: " << m_name << "\n";
    o << "kernel count: " << m_kernels.size() << "\n";
    for (int i = 0; i < (int)m_kernels.size(); ++i) {
        const unary_gfunc_kernel &k = m_kernels[i];
        o << "kernel " << i << "\n";
        o << "   " << k.m_out << " (" << k.m_params[0] << ")\n";
        o << "   func ptr: " << (const void *)k.m_kernel.kernel <<
                ", aux data: " << (const void *)(const dnd::AuxDataBase *)k.m_kernel.auxdata << "\n";
    }
    o << "------" << endl;
    return o.str();
}
