//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <algorithm>

#include <dnd/nodes/elwise_unary_kernel_node.hpp>
#include <dnd/nodes/elwise_binary_kernel_node.hpp>
#include <dnd/memblock/external_memory_block.hpp>

#include "elwise_gfunc_functions.hpp"
#include "ndarray_functions.hpp"
#include "utility_functions.hpp"
#include "ctypes_interop.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;

static void create_elwise_gfunc_kernel_from_ctypes(dnd::codegen_cache& cgcache, PyCFuncPtrObject *cfunc, dnd::gfunc::elwise_gfunc_kernel& out_kernel)
{
    dtype& returntype = out_kernel.m_returntype;
    vector<dtype> &paramtypes = out_kernel.m_paramtypes;
    get_ctypes_signature(cfunc, returntype, paramtypes);

    if (returntype.type_id() == void_type_id) {
        // TODO: May want support for this later, e.g. for print or other output functions
        throw std::runtime_error("Cannot construct a gfunc kernel from a ctypes function which returns void");
    }

    memory_block_ptr ctypes_memblock = make_external_memory_block(cfunc, &py_decref_function);

    if (paramtypes.size() == 1) {
        cgcache.codegen_unary_function_adapter(returntype, paramtypes[0], get_ctypes_calling_convention(cfunc),
                            *(void **)cfunc->b_ptr, ctypes_memblock.get(), out_kernel.m_unary_kernel);
    } else if (paramtypes.size() == 2) {
        cgcache.codegen_binary_function_adapter(returntype, paramtypes[0], paramtypes[1], get_ctypes_calling_convention(cfunc),
                            *(void **)cfunc->b_ptr, ctypes_memblock.get(), out_kernel.m_binary_kernel);
    } else {
        std::stringstream ss;
        ss << "gfunc kernels with " << paramtypes.size() << "parameters are not yet supported";
        throw std::runtime_error(ss.str());
    }
}

void pydnd::elwise_gfunc_add_kernel(dnd::gfunc::elwise_gfunc& gf, dnd::codegen_cache& cgcache, PyObject *kernel)
{
    if (PyObject_IsSubclass((PyObject *)Py_TYPE(kernel), ctypes.PyCFuncPtrType_Type)) {
        gfunc::elwise_gfunc_kernel egk;

        create_elwise_gfunc_kernel_from_ctypes(cgcache, (PyCFuncPtrObject *)kernel, egk);
        gf.add_kernel(egk);

        return;
    }

    throw std::runtime_error("Object could not be used as a gfunc kernel");
}

PyObject *pydnd::elwise_gfunc_call(dnd::gfunc::elwise_gfunc& gf, PyObject *args, PyObject *kwargs)
{
    Py_ssize_t nargs = PySequence_Size(args);

    // Convert the args into ndarrays, and get the value dtypes
    vector<ndarray> ndarray_args(nargs);
    vector<dtype> argtypes(nargs);
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        pyobject_ownref arg_obj(PySequence_GetItem(args, i));
        ndarray_init_from_pyobject(ndarray_args[i], arg_obj);
        argtypes[i] = ndarray_args[i].get_dtype().value_dtype();
    }

    const gfunc::elwise_gfunc_kernel *egk;
    egk = gf.find_matching_kernel(argtypes);

    if (egk == NULL) {
        std::stringstream ss;
        ss << gf.get_name() << ": could not find a gfunc kernel matching input argument types (";
        for (Py_ssize_t i = 0; i < nargs; ++i) {
            ss << argtypes[i];
            if (i != nargs - 1) {
                ss << ", ";
            }
        }
        ss << ")";
        throw std::runtime_error(ss.str());
    }

    if (nargs == 1) {
        ndarray result(make_elwise_unary_kernel_node_copy_kernel(
                    egk->m_returntype, ndarray_args[0].get_node(), egk->m_unary_kernel));
        pyobject_ownref result_obj(WNDArray_Type->tp_alloc(WNDArray_Type, 0));
        ((WNDArray *)result_obj.get())->v.swap(result);
        return result_obj.release();
    } else if (nargs == 2) {
        ndarray result(make_elwise_binary_kernel_node_copy_kernel(
                    egk->m_returntype, ndarray_args[0].get_node(), ndarray_args[1].get_node(), egk->m_binary_kernel));
        pyobject_ownref result_obj(WNDArray_Type->tp_alloc(WNDArray_Type, 0));
        ((WNDArray *)result_obj.get())->v.swap(result);
        return result_obj.release();
    } else {
        PyErr_SetString(PyExc_TypeError, "Elementwise gfuncs only support 1 or 2 arguments presently");
        return NULL;
    }
}
