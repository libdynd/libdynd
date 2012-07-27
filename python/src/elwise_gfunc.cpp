//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dnd/nodes/elwise_unary_kernel_node.hpp>
#include <dnd/nodes/elwise_binary_kernel_node.hpp>

#include "elwise_gfunc.hpp"
#include "ndarray_functions.hpp"
#include "utility_functions.hpp"
#include "ctypes_interop.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;

pydnd::elwise_gfunc_kernel::~elwise_gfunc_kernel()
{
    Py_XDECREF(m_pyobj);
}

pydnd::elwise_gfunc::~elwise_gfunc()
{
    for (size_t i = 0, i_end = m_blockrefs.size(); i != i_end; ++i) {
        memory_block_decref(m_blockrefs[i]);
    }
}

static void create_elwise_gfunc_kernel_from_ctypes(dnd::codegen_cache& cgcache, PyCFuncPtrObject *cfunc, elwise_gfunc_kernel& out_kernel)
{
    vector<dtype> &sig = out_kernel.m_sig;
    get_ctypes_signature(cfunc, sig);

    if (sig[0].type_id() == void_type_id) {
        // TODO: May want support for this later, e.g. for print or other output functions
        throw std::runtime_error("Cannot construct a gfunc kernel from a ctypes function which returns void");
    }

    out_kernel.m_pyobj = (PyObject *)cfunc;
    Py_INCREF(out_kernel.m_pyobj);

    if (sig.size() == 2) {
        cgcache.codegen_unary_function_adapter(sig[0], sig[1], get_ctypes_calling_convention(cfunc),
                            *(void **)cfunc->b_ptr, out_kernel.m_unary_kernel);
    } else if (sig.size() == 3) {
        cgcache.codegen_binary_function_adapter(sig[0], sig[1], sig[2], get_ctypes_calling_convention(cfunc),
                            *(void **)cfunc->b_ptr, out_kernel.m_binary_kernel);
    } else {
        std::stringstream ss;
        ss << "gfunc kernels with " << (sig.size() - 1) << "parameters are not yet supported";
        throw std::runtime_error(ss.str());
    }
}

void pydnd::elwise_gfunc::add_blockref(dnd::memory_block_data *blockref)
{
    if (find(m_blockrefs.begin(), m_blockrefs.end(), blockref) != m_blockrefs.end()) {
        m_blockrefs.push_back(blockref);
        memory_block_incref(blockref);
    }
}


void pydnd::elwise_gfunc::add_kernel(dnd::codegen_cache& cgcache, PyObject *kernel)
{
    if (PyObject_IsSubclass((PyObject *)Py_TYPE(kernel), ctypes.PyCFuncPtrType_Type)) {
        elwise_gfunc_kernel ugk;

        create_elwise_gfunc_kernel_from_ctypes(cgcache, (PyCFuncPtrObject *)kernel, ugk);
        m_kernels.push_back(elwise_gfunc_kernel());
        ugk.swap(m_kernels.back());

        add_blockref(cgcache.get_exec_memblock().get());
        return;
    }

    throw std::runtime_error("Object could not be used as a gfunc kernel");
}

PyObject *pydnd::elwise_gfunc::call(PyObject *args, PyObject *kwargs)
{
    Py_ssize_t nargs = PySequence_Size(args);
    if (nargs == 1) {
        pyobject_ownref arg0_obj(PySequence_GetItem(args, 0));
        ndarray arg0;
        ndarray_init_from_pyobject(arg0, arg0_obj);

        const dtype& dt0 = arg0.get_dtype().value_dtype();
        for (deque<elwise_gfunc_kernel>::size_type i = 0; i < m_kernels.size(); ++i) {
            const std::vector<dtype>& sig = m_kernels[i].m_sig;
            if (sig.size() == 2 && dt0 == sig[1]) {
                ndarray result(make_elwise_unary_kernel_node_copy_kernel(
                            sig[0], arg0.get_node(), m_kernels[i].m_unary_kernel));
                pyobject_ownref result_obj(WNDArray_Type->tp_alloc(WNDArray_Type, 0));
                ((WNDArray *)result_obj.get())->v.swap(result);
                return result_obj.release();
            }
        }

        std::stringstream ss;
        ss << "Could not find a gfunc kernel matching input dtype (" << dt0 << ")";
        throw std::runtime_error(ss.str());
    } else if (nargs == 2) {
        pyobject_ownref arg0_obj(PySequence_GetItem(args, 0));
        pyobject_ownref arg1_obj(PySequence_GetItem(args, 1));
        ndarray arg0, arg1;
        ndarray_init_from_pyobject(arg0, arg0_obj);
        ndarray_init_from_pyobject(arg1, arg1_obj);

        const dtype& dt0 = arg0.get_dtype().value_dtype();
        const dtype& dt1 = arg1.get_dtype().value_dtype();
        for (deque<elwise_gfunc_kernel>::size_type i = 0; i < m_kernels.size(); ++i) {
            const std::vector<dtype>& sig = m_kernels[i].m_sig;
            if (sig.size() == 3 && dt0 == sig[1] && dt1 == sig[2]) {
                ndarray result(make_elwise_binary_kernel_node_copy_kernel(
                            sig[0], arg0.get_node(), arg1.get_node(), m_kernels[i].m_binary_kernel));
                pyobject_ownref result_obj(WNDArray_Type->tp_alloc(WNDArray_Type, 0));
                ((WNDArray *)result_obj.get())->v.swap(result);
                return result_obj.release();
            }
        }

        std::stringstream ss;
        ss << "Could not find a gfunc kernel matching input dtypes (" << dt0 << ", " << dt1 << ")";
        throw std::runtime_error(ss.str());
    } else {
        PyErr_SetString(PyExc_TypeError, "Elementwise gfuncs only support 1 or 2 arguments presently");
        return NULL;
    }
}

std::string pydnd::elwise_gfunc::debug_dump() const
{
    std::stringstream o;
    o << "------ elwise_gfunc\n";
    o << "name: " << m_name << "\n";
    o << "kernel count: " << m_kernels.size() << "\n";
    for (deque<elwise_gfunc_kernel>::size_type i = 0; i < m_kernels.size(); ++i) {
        const elwise_gfunc_kernel &k = m_kernels[i];
        o << "kernel " << i << "\n";
        o << "   " << k.m_sig[0] << " (";
        for (size_t j = 1, j_end = k.m_sig.size(); j != j_end; ++j) {
            o << k.m_sig[j];
            if (j != j_end - 1) {
                o << ", ";
            }
        }
        o << ")\n";
        if (k.m_sig.size() == 2) {
            o << "unary aux data: " << (const void *)(const dnd::AuxDataBase *)k.m_unary_kernel.auxdata << "\n";
        } else if (k.m_sig.size() == 3) {
            o << "binary aux data: " << (const void *)(const dnd::AuxDataBase *)k.m_binary_kernel.auxdata << "\n";
        }
    }
    o << "------" << endl;
    return o.str();
}
