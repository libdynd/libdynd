//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dnd/nodes/elementwise_unary_kernel_node.hpp>

#include "unary_gfunc.hpp"
#include "ndarray_functions.hpp"
#include "utility_functions.hpp"
#include "ctypes_interop.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;

pydnd::unary_gfunc_kernel::~unary_gfunc_kernel()
{
    Py_XDECREF(m_pyobj);
}

pydnd::unary_gfunc::~unary_gfunc()
{
    for (size_t i = 0, i_end = m_blockrefs.size(); i != i_end; ++i) {
        memory_block_decref(m_blockrefs[i]);
    }
}

static void create_unary_gfunc_kernel_from_ctypes(dnd::codegen_cache& cgcache, PyCFuncPtrObject *cfunc, unary_gfunc_kernel& out_kernel)
{
    vector<dtype> sig;
    get_ctypes_signature(cfunc, sig);

    if (sig[0].type_id() == void_type_id) {
        // TODO: May want support for this later, e.g. for print or other output functions
        throw std::runtime_error("Cannot construct a gfunc kernel from a ctypes function which returns void");
    }

    if (sig.size() != 2) {
        std::stringstream ss;
        ss << "Only unary gfunc kernels are currently supported, provided gfunc has " << (sig.size() - 1);
        throw std::runtime_error(ss.str());
    }

    out_kernel.m_out = sig[0];
    out_kernel.m_params[0] = sig[1];

    calling_convention_t cc = get_ctypes_calling_convention(cfunc);

    out_kernel.m_kernel.specializations = cgcache.codegen_unary_function_adapter(sig[0], sig[1], cc);

    // Use the function pointer as the raw auxiliary data
    make_raw_auxiliary_data(out_kernel.m_kernel.auxdata, *(uintptr_t *)cfunc->b_ptr);
}

void pydnd::unary_gfunc::add_blockref(dnd::memory_block_data *blockref)
{
    if (find(m_blockrefs.begin(), m_blockrefs.end(), blockref) != m_blockrefs.end()) {
        m_blockrefs.push_back(blockref);
        memory_block_incref(blockref);
    }
}


void pydnd::unary_gfunc::add_kernel(dnd::codegen_cache& cgcache, PyObject *kernel)
{
    if (PyObject_IsSubclass((PyObject *)Py_TYPE(kernel), ctypes.PyCFuncPtrType_Type)) {
        unary_gfunc_kernel ugk;
        create_unary_gfunc_kernel_from_ctypes(cgcache, (PyCFuncPtrObject *)kernel, ugk);
        ugk.m_pyobj = kernel;
        m_kernels.push_back(unary_gfunc_kernel());
        ugk.swap(m_kernels.back());
        add_blockref(cgcache.get_exec_memblock().get());
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
