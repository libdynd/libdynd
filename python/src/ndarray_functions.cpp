//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "ndarray_functions.hpp"
#include "ndarray_from_py.hpp"
#include "dtype_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

#include <dnd/dtypes/string_dtype.hpp>
#include <dnd/memblock/external_memory_block.hpp>
#include <dnd/nodes/scalar_node.hpp>
#include <dnd/nodes/groupby_node.hpp>
#include <dnd/ndarray_arange.hpp>
#include <dnd/dtype_promotion.hpp>

using namespace std;
using namespace dnd;
using namespace pydnd;

PyTypeObject *pydnd::WNDArray_Type;

void pydnd::init_w_ndarray_typeobject(PyObject *type)
{
    WNDArray_Type = (PyTypeObject *)type;
}

dnd::ndarray pydnd::ndarray_vals(const dnd::ndarray& n)
{
    return n.vals();
}

dnd::ndarray pydnd::ndarray_eval_copy(const dnd::ndarray& n, PyObject* access_flags, const eval::eval_context *ectx)
{
    if (access_flags == Py_None) {
        return n.eval_copy(ectx);
    } else {
        return n.eval_copy(ectx, pyarg_access_flags(access_flags));
    }
}

static irange pyobject_as_irange(PyObject *index)
{
    if (PySlice_Check(index)) {
        irange result;
        PySliceObject *slice = (PySliceObject *)index;
        if (slice->start != Py_None) {
            result.set_start(pyobject_as_index(slice->start));
        }
        if (slice->stop != Py_None) {
            result.set_finish(pyobject_as_index(slice->stop));
        }
        if (slice->step != Py_None) {
            result.set_step(pyobject_as_index(slice->step));
        }
        return result;
    } else {
        return irange(pyobject_as_index(index));
    }
}

dnd::ndarray pydnd::ndarray_getitem(const dnd::ndarray& n, PyObject *subscript)
{
    // Convert the pyobject into an array of iranges
    intptr_t size;
    shortvector<irange> indices;
    if (!PyTuple_Check(subscript)) {
        // A single subscript
        size = 1;
        indices.init(1);
        indices[0] = pyobject_as_irange(subscript);
    } else {
        size = PyTuple_GET_SIZE(subscript);
        // Tuple of subscripts
        indices.init(size);
        for (Py_ssize_t i = 0; i < size; ++i) {
            indices[i] = pyobject_as_irange(PyTuple_GET_ITEM(subscript, i));
        }
    }

    // Do an indexing operation
    return n.index(size, indices.get());
}

ndarray pydnd::ndarray_arange(PyObject *start, PyObject *stop, PyObject *step)
{
    ndarray start_nd, stop_nd, step_nd;
    if (start != Py_None) {
        ndarray_init_from_pyobject(start_nd, start);
    } else {
        start_nd = 0;
    }
    ndarray_init_from_pyobject(stop_nd, stop);
    if (step != Py_None) {
        ndarray_init_from_pyobject(step_nd, step);
    } else {
        step_nd = 1;
    }
    
    dtype dt = promote_dtypes_arithmetic(start_nd.get_dtype(),
            promote_dtypes_arithmetic(stop_nd.get_dtype(), step_nd.get_dtype()));
    
    start_nd = start_nd.as_dtype(dt, assign_error_none).vals();
    stop_nd = stop_nd.as_dtype(dt, assign_error_none).vals();
    step_nd = step_nd.as_dtype(dt, assign_error_none).vals();

    if (start_nd.get_ndim() > 0 || stop_nd.get_ndim() > 0 || step_nd.get_ndim()) {
        throw runtime_error("dnd::arange should only be called with scalar parameters");
    }

    return arange(dt, start_nd.get_readonly_originptr(),
            stop_nd.get_readonly_originptr(),
            step_nd.get_readonly_originptr());
}

dnd::ndarray pydnd::ndarray_linspace(PyObject *start, PyObject *stop, PyObject *count)
{
    ndarray start_nd, stop_nd;
    intptr_t count_val = pyobject_as_index(count);
    ndarray_init_from_pyobject(start_nd, start);
    ndarray_init_from_pyobject(stop_nd, stop);
    dtype dt = promote_dtypes_arithmetic(start_nd.get_dtype(), stop_nd.get_dtype());
    // Make sure it's at least floating point
    if (dt.kind() == bool_kind || dt.kind() == int_kind || dt.kind() == uint_kind) {
        dt = make_dtype<double>();
    }
    start_nd = start_nd.as_dtype(dt, assign_error_none).vals();
    stop_nd = stop_nd.as_dtype(dt, assign_error_none).vals();

    if (start_nd.get_ndim() > 0 || stop_nd.get_ndim() > 0) {
        throw runtime_error("dnd::linspace should only be called with scalar parameters");
    }

    return linspace(dt, start_nd.get_readonly_originptr(), stop_nd.get_readonly_originptr(), count_val);
}

dnd::ndarray pydnd::ndarray_groupby(const dnd::ndarray& data, const dnd::ndarray& by, const dnd::dtype& groups)
{
    return ndarray(make_groupby_node(data.get_node(), by.get_node(), groups));
}
