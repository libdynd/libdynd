//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dnd/ndarray.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/exceptions.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/diagnostics.hpp>

#include "ndarray_expr_node_instances.hpp"

using namespace std;
using namespace dnd;

// Node factory functions

ndarray_expr_node_ptr dnd::make_strided_array_expr_node(
            const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr,
            const memory_block_ref& memblock)
{
    // TODO: Add a multidimensional DND_ASSERT_ALIGNED check here
    return ndarray_expr_node_ptr(new strided_array_expr_node(dt, ndim,
                                        shape, strides, originptr, memblock));
}

ndarray_expr_node_ptr dnd::apply_index_to_node(ndarray_expr_node *node,
                                int nindex, const irange *indices, bool allow_in_place)
{
    // Validate the number of indices
    if (nindex > node->get_ndim()) {
        throw too_many_indices(nindex, node->get_ndim());
    }

    int ndim = node->get_ndim();
    const intptr_t *node_shape = node->get_shape();

    shortvector<bool> remove_axis(ndim);
    shortvector<intptr_t> start_index(ndim);
    shortvector<intptr_t> index_strides(ndim);
    shortvector<intptr_t> shape(ndim);

    // Convert the indices into the form used by the apply_linear_index function
    int new_i = 0;
    for (int i = 0; i < nindex; ++i) {
        intptr_t step = indices[i].step();
        intptr_t node_shape_i = node_shape[i];
        if (step == 0) {
            // A single index
            intptr_t idx = indices[i].start();
            if (idx >= 0) {
                if (idx < node_shape_i) {
                    // Regular single index
                    remove_axis[i] = true;
                    start_index[i] = idx;
                    index_strides[i] = 0;
                    shape[i] = 1;
                } else {
                    throw index_out_of_bounds(idx, i, ndim, node_shape);
                }
            } else if (idx >= -node_shape_i) {
                // Python style negative single index
                remove_axis[i] = true;
                start_index[i] = idx + node_shape_i;
                index_strides[i] = 0;
                shape[i] = 1;
            } else {
                throw index_out_of_bounds(idx, i, ndim, node_shape);
            }
        } else if (step > 0) {
            // A range with a positive step
            intptr_t start = indices[i].start();
            if (start >= 0) {
                if (start < node_shape_i) {
                    // Starts with a positive index
                } else {
                    throw irange_out_of_bounds(indices[i], i, ndim, node_shape);
                }
            } else if (start >= -node_shape_i) {
                // Starts with Python style negative index
                start += node_shape_i;
            } else if (start == std::numeric_limits<intptr_t>::min()) {
                // Signal for "from the beginning"
                start = 0;
            } else {
                throw irange_out_of_bounds(indices[i], i, ndim, node_shape);
            }

            intptr_t end = indices[i].finish();
            if (end >= 0) {
                if (end <= node_shape_i) {
                    // Ends with a positive index, or the end of the array
                } else if (end == std::numeric_limits<intptr_t>::max()) {
                    // Signal for "until the end"
                    end = node_shape_i;
                } else {
                    throw irange_out_of_bounds(indices[i], i, ndim, node_shape);
                }
            } else if (end >= -node_shape_i) {
                // Ends with a Python style negative index
                end += node_shape_i;
            } else {
                throw irange_out_of_bounds(indices[i], i, ndim, node_shape);
            }

            intptr_t size = end - start;
            if (size > 0) {
                if (step == 1) {
                    // Simple range
                    remove_axis[i] = false;
                    start_index[i] = start;
                    index_strides[i] = 1;
                    shape[i] = size;
                } else {
                    // Range with a stride
                    remove_axis[i] = false;
                    start_index[i] = start;
                    index_strides[i] = step;
                    shape[i] = (size + step - 1) / step;
                }
            } else {
                // Empty slice
                remove_axis[i] = false;
                start_index[i] = 0;
                index_strides[i] = 1;
                shape[i] = 0;
            }
        } else {
            // A range with a negative step
            intptr_t start = indices[i].start();
            if (start >= 0) {
                if (start < node_shape_i) {
                    // Starts with a positive index
                } else {
                    throw irange_out_of_bounds(indices[i], i, ndim, node_shape);
                }
            } else if (start >= -node_shape_i) {
                // Starts with Python style negative index
                start += node_shape_i;
            } else if (start == std::numeric_limits<intptr_t>::min()) {
                // Signal for "from the beginning" (which means the last element)
                start = node_shape_i - 1;
            } else {
                throw irange_out_of_bounds(indices[i], i, ndim, node_shape);
            }

            intptr_t end = indices[i].finish();
            if (end >= 0) {
                if (end < node_shape_i) {
                    // Ends with a positive index, or the end of the array
                } else if (end == std::numeric_limits<intptr_t>::max()) {
                    // Signal for "until the end" (which means towards index 0 of the data)
                    end = -1;
                } else {
                    throw irange_out_of_bounds(indices[i], i, ndim, node_shape);
                }
            } else if (end >= -node_shape_i) {
                // Ends with a Python style negative index
                end += node_shape_i;
            } else {
                throw irange_out_of_bounds(indices[i], i, ndim, node_shape);
            }

            intptr_t size = start - end;
            if (size > 0) {
                if (step == -1) {
                    // Simple range
                    remove_axis[i] = false;
                    start_index[i] = start;
                    index_strides[i] = -1;
                    shape[i] = size;
                } else {
                    // Range with a stride
                    remove_axis[i] = false;
                    start_index[i] = start;
                    index_strides[i] = step;
                    shape[i] = (size + (-step) - 1) / (-step);
                }
            } else {
                // Empty slice
                remove_axis[i] = false;
                start_index[i] = 0;
                index_strides[i] = 1;
                shape[i] = 0;
            }
        }
    }

    // Indexing applies to the left, fill the rest with no-op indexing
    for (int i = nindex; i < ndim; ++i) {
        remove_axis[i] = false;
        start_index[i] = 0;
        index_strides[i] = 1;
        shape[i] = node_shape[i];
    }

    return node->apply_linear_index(ndim, remove_axis.get(), start_index.get(), index_strides.get(), shape.get(), allow_in_place);
}

ndarray_expr_node_ptr dnd::apply_integer_index_to_node(ndarray_expr_node *node,
                                int axis, intptr_t idx, bool allow_in_place)
{
    int ndim = node->get_ndim();

    if (axis < 0 || axis >= ndim) {
        throw axis_out_of_bounds(axis, 0, ndim);
    }
    int shape_axis = node->get_shape()[axis];

    if (idx >= 0) {
        if (idx < shape_axis) {
            // Normal positive index
        } else {
            throw index_out_of_bounds(idx, idx, ndim, node->get_shape());
        }
    } else if (idx >= -shape_axis) {
        // Python style negative index
        idx += shape_axis;
    } else {
        throw index_out_of_bounds(idx, idx, ndim, node->get_shape());
    }

    shortvector<bool> remove_axis(ndim);
    shortvector<intptr_t> start_index(ndim);
    shortvector<intptr_t> index_strides(ndim);

    for (int i = 0; i < ndim; ++i) {
        remove_axis[i] = false;
    }
    remove_axis[axis] = true;

    for (int i = 0; i < ndim; ++i) {
        start_index[i] = 0;
    }
    start_index[axis] = idx;

    for (int i = 0; i < ndim; ++i) {
        index_strides[i] = 1;
    }
    index_strides[axis] = 0;

    return node->apply_linear_index(ndim, remove_axis.get(), start_index.get(), index_strides.get(), node->get_shape(), allow_in_place);
}
