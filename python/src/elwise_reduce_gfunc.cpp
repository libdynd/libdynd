//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dnd/nodes/elwise_reduce_kernel_node.hpp>

#include "elwise_reduce_gfunc.hpp"
#include "ndarray_functions.hpp"
#include "utility_functions.hpp"
#include "ctypes_interop.hpp"

using namespace std;
using namespace dnd;
using namespace pydnd;

pydnd::elwise_reduce_gfunc_kernel::~elwise_reduce_gfunc_kernel()
{
    Py_XDECREF(m_pyobj);
}

pydnd::elwise_reduce_gfunc::~elwise_reduce_gfunc()
{
    for (size_t i = 0, i_end = m_blockrefs.size(); i != i_end; ++i) {
        memory_block_decref(m_blockrefs[i]);
    }
}

static void create_elwise_reduce_gfunc_kernel_from_ctypes(dnd::codegen_cache& cgcache,
            PyCFuncPtrObject *cfunc, bool associative, bool commutative, const ndarray& identity, elwise_reduce_gfunc_kernel& out_kernel)
{
    vector<dtype> &sig = out_kernel.m_sig;
    get_ctypes_signature(cfunc, sig);

    out_kernel.m_associative = associative;
    out_kernel.m_commutative = commutative;

    out_kernel.m_pyobj = (PyObject *)cfunc;
    Py_INCREF(out_kernel.m_pyobj);

    if (sig[0].type_id() == void_type_id) {
        // TODO: Should support this if the kernel is flagged as commutative,
        //       in which case the first parameter must be an inout pointer parameter
        throw std::runtime_error("Cannot construct a gfunc reduce kernel from a single ctypes function which returns void");
    }

    if (sig.size() == 2) {
        if (!commutative) {
            throw runtime_error("To use an in-place reduction kernel, the kernel must either be commutative, or"
                        " both left and right associative variants must be provided");
        }

    } else if (sig.size() == 3) {
        if (sig[0] != sig[1] || sig[0] != sig[2]) {
            std::stringstream ss;
            ss << "A binary reduction kernel must have all three types equal.";
            ss << " Provided signature " << sig[0] << " (" << sig[1] << ", " << sig[2] << ")";
            throw std::runtime_error(ss.str());
        }
        cgcache.codegen_left_associative_binary_reduce_function_adapter(sig[0], get_ctypes_calling_convention(cfunc),
                            *(void **)cfunc->b_ptr, out_kernel.m_left_associative_reduction_kernel);
        if (!commutative) {
            cgcache.codegen_right_associative_binary_reduce_function_adapter(sig[0], get_ctypes_calling_convention(cfunc),
                                *(void **)cfunc->b_ptr, out_kernel.m_right_associative_reduction_kernel);
        }

        // The adapted reduction signature has just two types
        sig.pop_back();
    } else {
        std::stringstream ss;
        ss << "A single function provided as a gfunc reduce kernel must be binary, the provided one has " << (sig.size() - 1);
        throw std::runtime_error(ss.str());
    }

    // If an identity is provided, get an immutable version of it as the reduction dtype
    if (identity.get_node().get() != NULL) {
        out_kernel.m_identity = identity.as_dtype(sig[0]).eval_immutable();
    } else {
        out_kernel.m_identity = ndarray();
    }
}

void pydnd::elwise_reduce_gfunc::add_blockref(dnd::memory_block_data *blockref)
{
    if (find(m_blockrefs.begin(), m_blockrefs.end(), blockref) != m_blockrefs.end()) {
        m_blockrefs.push_back(blockref);
        memory_block_incref(blockref);
    }
}



void pydnd::elwise_reduce_gfunc::add_kernel(dnd::codegen_cache& cgcache, PyObject *kernel,
                            bool associative, bool commutative, const ndarray& identity)
{
    if (PyObject_IsSubclass((PyObject *)Py_TYPE(kernel), ctypes.PyCFuncPtrType_Type)) {
        elwise_reduce_gfunc_kernel ugk;

        create_elwise_reduce_gfunc_kernel_from_ctypes(cgcache, (PyCFuncPtrObject *)kernel, associative, commutative, identity, ugk);
        m_kernels.push_back(elwise_reduce_gfunc_kernel());
        ugk.swap(m_kernels.back());

        add_blockref(cgcache.get_exec_memblock().get());
        return;
    }

    throw std::runtime_error("Object could not be used as a gfunc kernel");
}

PyObject *pydnd::elwise_reduce_gfunc::call(PyObject *args, PyObject *kwargs)
{
    Py_ssize_t nargs = PySequence_Size(args);
    if (nargs == 1) {
        pyobject_ownref arg0_obj(PySequence_GetItem(args, 0));
        ndarray arg0;
        ndarray_init_from_pyobject(arg0, arg0_obj);

        shortvector<dnd_bool> reduce_axes(arg0.get_ndim());

        // axis=[integer OR tuple of integers]
        int axis_count = pyarg_axis_argument(PyDict_GetItemString(kwargs, "axis"), arg0.get_ndim(), reduce_axes.get());

        // associate=['left' OR 'right']
        bool rightassoc = pyarg_strings_to_int(PyDict_GetItemString(kwargs, "associate"), "associate", 0,
                            "left", 0,
                            "right", 1) == 1;

        // keepdims
        bool keepdims = pyarg_bool(PyDict_GetItemString(kwargs, "keepdims"), "keepdims", false);

        const dtype& dt0 = arg0.get_dtype();
        for (deque<elwise_reduce_gfunc_kernel>::size_type i = 0; i < m_kernels.size(); ++i) {
            const std::vector<dtype>& sig = m_kernels[i].m_sig;
            if (sig.size() == 2 && dt0 == sig[1]) {
                if (axis_count > 1 && !m_kernels[i].m_commutative) {
                    stringstream ss;
                    ss << "Cannot call non-commutative reduce gfunc " << m_name << " with more than one axis";
                    throw runtime_error(ss.str());
                }
                ndarray result(make_elwise_reduce_kernel_node_copy_kernel(
                            sig[0], arg0.get_node(), reduce_axes.get(), rightassoc, keepdims, m_kernels[i].m_identity.get_node(),
                            (!rightassoc || m_kernels[i].m_commutative) ? m_kernels[i].m_left_associative_reduction_kernel :
                                    m_kernels[i].m_right_associative_reduction_kernel));
                pyobject_ownref result_obj(WNDArray_Type->tp_alloc(WNDArray_Type, 0));
                ((WNDArray *)result_obj.get())->v.swap(result);
                return result_obj.release();
            }
        }

        std::stringstream ss;
        ss << "Could not find a gfunc kernel matching input dtype (" << dt0 << ")";
        throw std::runtime_error(ss.str());
    } else {
        PyErr_SetString(PyExc_TypeError, "Elementwise reduction gfuncs only support 1 argument");
        return NULL;
    }
}

std::string pydnd::elwise_reduce_gfunc::debug_dump() const
{
    std::stringstream o;
    o << "------ elwise_reduce_gfunc\n";
    o << "name: " << m_name << "\n";
    o << "kernel count: " << m_kernels.size() << "\n";
    for (deque<elwise_reduce_gfunc_kernel>::size_type i = 0; i < m_kernels.size(); ++i) {
        const elwise_reduce_gfunc_kernel &k = m_kernels[i];
        o << "kernel " << i << "\n";
        o << " signature: " << k.m_sig[0] << " (";
        for (size_t j = 1, j_end = k.m_sig.size(); j != j_end; ++j) {
            o << k.m_sig[j];
            if (j != j_end - 1) {
                o << ", ";
            }
        }
        o << ")\n";
        if (k.m_left_associative_reduction_kernel.kernel != NULL) {
            o << " left associative kernel aux data: " << (const void *)(const dnd::AuxDataBase *)k.m_left_associative_reduction_kernel.auxdata << "\n";
        }
        if (k.m_right_associative_reduction_kernel.kernel != NULL) {
            o << " right associative kernel aux data: " << (const void *)(const dnd::AuxDataBase *)k.m_right_associative_reduction_kernel.auxdata << "\n";
        }
        o << " reduction identity:\n";
        k.m_identity.debug_dump(o, "  ");
    }
    o << "------" << endl;
    return o.str();
}
