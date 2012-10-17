//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <string>

#include <dynd/shape_tools.hpp>
#include <dynd/ndarray.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/nodes/ndarray_node.hpp>

using namespace std;
using namespace dynd;

static void print_ndarray_access_flags(std::ostream& o, int access_flags)
{
    if (access_flags & read_access_flag) {
        o << "read ";
    }
    if (access_flags & write_access_flag) {
        o << "write ";
    }
    if (access_flags & immutable_access_flag) {
        o << "immutable ";
    }
}


const intptr_t *dynd::ndarray_node::get_strides() const
{
    throw std::runtime_error("cannot get strides from an ndarray node which is not strided");
}

void dynd::ndarray_node::get_right_broadcast_strides(int ndim, intptr_t *out_strides) const
{
    int node_ndim = get_ndim();
    if (ndim == node_ndim) {
        memcpy(out_strides, get_strides(), node_ndim * sizeof(intptr_t));
    } else {
        memset(out_strides, 0, (ndim - node_ndim) * sizeof(intptr_t));
        memcpy(out_strides + (ndim - node_ndim), get_strides(), node_ndim * sizeof(intptr_t));
    }
}

const char *dynd::ndarray_node::get_readonly_originptr() const
{
    throw std::runtime_error("cannot get a readonly originptr from an ndarray node which is not strided");
}

char *dynd::ndarray_node::get_readwrite_originptr() const
{
    throw std::runtime_error("cannot get a readwrite originptr from an ndarray node which is not strided");
}

void dynd::ndarray_node::get_unary_operation(intptr_t, intptr_t, kernel_instance<unary_operation_t>&) const
{
    throw std::runtime_error("get_unary_operation is only valid for "
                             "unary nodes which provide an implementation");
}

void dynd::ndarray_node::get_unary_specialization_operation(unary_specialization_kernel_instance& DYND_UNUSED(out_kernel)) const
{
    throw std::runtime_error("unary_specialization_kernel_instance is only valid for "
                             "unary nodes which provide an implementation");
}


void dynd::ndarray_node::get_binary_operation(intptr_t, intptr_t, intptr_t,
                        const eval::eval_context *DYND_UNUSED(ectx),
                        kernel_instance<binary_operation_t>&) const
{
    throw std::runtime_error("get_binary_operation is only valid for "
                             "binary nodes which provide an implementation");
}

memory_block_ptr dynd::ndarray_node::get_data_memory_block()
{
    return memory_block_ptr();
}

static void print_node_category(ostream& o, ndarray_node_category cat)
{
    switch (cat) {
        case strided_array_node_category:
            o << "strided_array_node_category";
            break;
        case elwise_node_category:
            o << "elwise_node_category";
            break;
        case elwise_reduce_node_category:
            o << "elwise_reduce_node_category";
            break;
        case groupby_node_category:
            o << "groupby_node_category";
            break;
        case arbitrary_node_category:
            o << "arbitrary_node_category";
            break;
        default:
            o << "unknown category (" << (int)cat << ")";
            break;
    }
}

void dynd::ndarray_node::debug_dump(ostream& o, const string& indent) const
{
    o << indent << "(\"" << node_name() << "\",\n";

    o << indent << " dtype: " << get_dtype() << "\n";
    o << indent << " ndim: " << get_ndim() << "\n";
    o << indent << " shape: ";
    print_shape(o, get_ndim(), get_shape());
    o << "\n";
    o << indent << " node category: ";
    print_node_category(o, get_category());
    o << "\n";
    o << indent << " access flags: ";
    print_ndarray_access_flags(o, get_access_flags());
    o << "\n";
    debug_dump_extra(o, indent);

    if (get_nop() > 0) {
        o << indent << " nop: " << get_nop() << "\n";
        for (int i = 0; i < get_nop(); ++i) {
            o << indent << " operand " << i << ":\n";
            get_opnode(i)->debug_dump(o, indent + "  ");
        }
    }

    o << indent << ")\n";
}

void dynd::ndarray_node::debug_dump_extra(ostream&, const string&) const
{
    // Default is no extra information
}

ndarray_node_ptr dynd::apply_index_to_node(const ndarray_node_ptr& node,
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

ndarray_node_ptr dynd::apply_integer_index_to_node(const ndarray_node_ptr& node,
                                int axis, intptr_t idx, bool allow_in_place)
{
    int ndim = node->get_ndim();

    if (axis < 0 || axis >= ndim) {
        throw axis_out_of_bounds(axis, ndim);
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
