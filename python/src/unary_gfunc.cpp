//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/nodes/elementwise_unary_kernel_node.hpp>

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
            cdecl_func_ptr_t kfunc = reinterpret_cast<cdecl_func_ptr_t>(get_raw_auxiliary_data(auxdata)&(~1));

            for (intptr_t i = 0; i < count; ++i) {
                *(S *)dst = kfunc(*(const T *)src);
                dst += dst_stride;
                src += src_stride;
            }
        }

        static specialized_unary_operation_table_t cdecl_kerneltable;

#if defined(_WIN32) && !defined(_M_X64)
        typedef S (__stdcall *win32_stdcall_func_ptr_t)(T);

        static void win32_stdcall_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                                    intptr_t count, const AuxDataBase *auxdata)
        {
            win32_stdcall_func_ptr_t kfunc = reinterpret_cast<win32_stdcall_func_ptr_t>(get_raw_auxiliary_data(auxdata)&(~1));

            for (intptr_t i = 0; i < count; ++i) {
                *(S *)dst = kfunc(*(const T *)src);
                dst += dst_stride;
                src += src_stride;
            }
        }

        static specialized_unary_operation_table_t win32_stdcall_kerneltable;
#endif // 32-bit Windows
    }; // struct unary_kernels

    template<typename S, typename T>
    specialized_unary_operation_table_t unary_kernels<S, T>::cdecl_kerneltable = {
        &unary_kernels<S, T>::cdecl_kernel,
        &unary_kernels<S, T>::cdecl_kernel,
        &unary_kernels<S, T>::cdecl_kernel,
        &unary_kernels<S, T>::cdecl_kernel
        };

#if defined(_WIN32) && !defined(_M_X64)
    template<typename S, typename T>
    specialized_unary_operation_table_t unary_kernels<S, T>::win32_stdcall_kerneltable = {
        &unary_kernels<S, T>::win32_stdcall_kernel,
        &unary_kernels<S, T>::win32_stdcall_kernel,
        &unary_kernels<S, T>::win32_stdcall_kernel,
        &unary_kernels<S, T>::win32_stdcall_kernel
        };
#endif // 32-bit Windows
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
        pyobject_ownref element(PySequence_GetItem(argtypes, i));
        out_kernel.m_params[i] = dtype_from_ctypes_cdatatype(element);
    }

    if (out_kernel.m_out != out_kernel.m_params[0]) {
        throw std::runtime_error("Only gfunc kernels with matching input and output are currently supported");
    }

    calling_convention_t cc = get_ctypes_calling_convention(cfunc);
    //cout << "calling convention is " << cc << endl;
#if defined(_WIN32) && !defined(_M_X64)
    if (cc == win32_stdcall_callconv) {
        switch (out_kernel.m_params[0].type_id()) {
            case int8_type_id:
                out_kernel.m_kernel.specializations = unary_kernels<int8_t, int8_t>::win32_stdcall_kerneltable;
                break;
            case int16_type_id:
                out_kernel.m_kernel.specializations = unary_kernels<int16_t, int16_t>::win32_stdcall_kerneltable;
                break;
            case int32_type_id:
                out_kernel.m_kernel.specializations = unary_kernels<int32_t, int32_t>::win32_stdcall_kerneltable;
                break;
            case int64_type_id:
                out_kernel.m_kernel.specializations = unary_kernels<int64_t, int64_t>::win32_stdcall_kerneltable;
                break;
            case uint8_type_id:
                out_kernel.m_kernel.specializations = unary_kernels<uint8_t, int8_t>::win32_stdcall_kerneltable;
                break;
            case uint16_type_id:
                out_kernel.m_kernel.specializations = unary_kernels<uint16_t, uint16_t>::win32_stdcall_kerneltable;
                break;
            case uint32_type_id:
                out_kernel.m_kernel.specializations = unary_kernels<uint32_t, uint32_t>::win32_stdcall_kerneltable;
                break;
            case uint64_type_id:
                out_kernel.m_kernel.specializations = unary_kernels<uint64_t, uint64_t>::win32_stdcall_kerneltable;
                break;
            case float32_type_id:
                out_kernel.m_kernel.specializations = unary_kernels<float, float>::win32_stdcall_kerneltable;
                break;
            case float64_type_id:
                out_kernel.m_kernel.specializations = unary_kernels<double, double>::win32_stdcall_kerneltable;
                break;
            default:
                throw std::runtime_error("Couldn't construct a kernel for the ctypes function prototype");
        }
    }
#endif // 32-bit Windows

    switch (out_kernel.m_params[0].type_id()) {
        case int8_type_id:
            out_kernel.m_kernel.specializations = unary_kernels<int8_t, int8_t>::cdecl_kerneltable;
            break;
        case int16_type_id:
            out_kernel.m_kernel.specializations = unary_kernels<int16_t, int16_t>::cdecl_kerneltable;
            break;
        case int32_type_id:
            out_kernel.m_kernel.specializations = unary_kernels<int32_t, int32_t>::cdecl_kerneltable;
            break;
        case int64_type_id:
            out_kernel.m_kernel.specializations = unary_kernels<int64_t, int64_t>::cdecl_kerneltable;
            break;
        case uint8_type_id:
            out_kernel.m_kernel.specializations = unary_kernels<uint8_t, int8_t>::cdecl_kerneltable;
            break;
        case uint16_type_id:
            out_kernel.m_kernel.specializations = unary_kernels<uint16_t, uint16_t>::cdecl_kerneltable;
            break;
        case uint32_type_id:
            out_kernel.m_kernel.specializations = unary_kernels<uint32_t, uint32_t>::cdecl_kerneltable;
            break;
        case uint64_type_id:
            out_kernel.m_kernel.specializations = unary_kernels<uint64_t, uint64_t>::cdecl_kerneltable;
            break;
        case float32_type_id:
            out_kernel.m_kernel.specializations = unary_kernels<float, float>::cdecl_kerneltable;
            break;
        case float64_type_id:
            out_kernel.m_kernel.specializations = unary_kernels<double, double>::cdecl_kerneltable;
            break;
        default:
            throw std::runtime_error("Couldn't construct a kernel for the ctypes function prototype");
    }

    // Use the function pointer as the raw auxiliary data
    make_raw_auxiliary_data(out_kernel.m_kernel.auxdata, *(uintptr_t *)cfunc->b_ptr);
}

void pydnd::unary_gfunc::add_kernel(PyObject *kernel)
{
    if (PyObject_IsSubclass((PyObject *)Py_TYPE(kernel), ctypes.PyCFuncPtrType_Type)) {
        unary_gfunc_kernel ugk;
        create_unary_gfunc_kernel_from_ctypes((PyCFuncPtrObject *)kernel, ugk);
        ugk.m_pyobj = kernel;
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

    const dtype& dt = arg0.get_dtype();
    for (deque<unary_gfunc_kernel>::size_type i = 0; i < m_kernels.size(); ++i) {
        if (dt == m_kernels[i].m_params[0]) {
            ndarray result(make_elementwise_unary_kernel_node_copy_kernel(
                        m_kernels[i].m_out, arg0.get_expr_tree(), m_kernels[i].m_kernel));
            pyobject_ownref result_obj(WNDArray_Type->tp_alloc(WNDArray_Type, 0));
            ((WNDArray *)result_obj.get())->v.swap(result);
            return result_obj.release();
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
    for (deque<unary_gfunc_kernel>::size_type i = 0; i < m_kernels.size(); ++i) {
        const unary_gfunc_kernel &k = m_kernels[i];
        o << "kernel " << i << "\n";
        o << "   " << k.m_out << " (" << k.m_params[0] << ")\n";
        o << "aux data: " << (const void *)(const dnd::AuxDataBase *)k.m_kernel.auxdata << "\n";
    }
    o << "------" << endl;
    return o.str();
}
