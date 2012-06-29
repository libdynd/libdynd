//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <string>

#include <dnd/ndarray.hpp>
#include <dnd/exceptions.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/nodes/ndarray_node.hpp>
#include <dnd/nodes/elementwise_binary_kernel_node.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

using namespace std;
using namespace dnd;

void dnd::ndarray_node::as_readwrite_data_and_strides(int ndim, char ** DND_UNUSED(out_data),
                                                intptr_t * DND_UNUSED(out_strides)) const
{
    throw std::runtime_error("as_readwrite_data_and_strides is only valid for "
                             "nodes with an expr_node_strided_array category");
}

void dnd::ndarray_node::as_readonly_data_and_strides(int ndim, char const ** DND_UNUSED(out_data),
                                                intptr_t * DND_UNUSED(out_strides)) const
{
    throw std::runtime_error("as_readonly_data_and_strides is only valid for "
                             "nodes with an expr_node_strided_array category");
}

void dnd::ndarray_node::get_nullary_operation(intptr_t, kernel_instance<nullary_operation_t>&) const
{
    throw std::runtime_error("get_nullary_operation is only valid for "
                             "generator nodes which provide an implementation");
}

void dnd::ndarray_node::get_unary_operation(intptr_t, intptr_t, kernel_instance<unary_operation_t>&) const
{
    throw std::runtime_error("get_unary_operation is only valid for "
                             "unary nodes which provide an implementation");
}

void dnd::ndarray_node::get_binary_operation(intptr_t, intptr_t, intptr_t, kernel_instance<binary_operation_t>&) const
{
    throw std::runtime_error("get_binary_operation is only valid for "
                             "binary nodes which provide an implementation");
}

ndarray_node_ref dnd::ndarray_node::evaluate()
{
    const dtype& dt = get_dtype();

    switch (get_category()) {
        case strided_array_node_category: {
            // Evaluate any expression dtype as well
            if (dt.kind() == expression_kind) {
                ndarray_node_ref result;
                raw_ndarray_iter<1,1> iter(m_ndim, m_shape.get(), dt.value_dtype(), result, this);

                intptr_t innersize = iter.innersize();
                intptr_t dst_stride = iter.innerstride<0>();
                intptr_t src0_stride = iter.innerstride<1>();
                unary_specialization_kernel_instance operation;
                get_dtype_assignment_kernel(dt.value_dtype(), dt, assign_error_none, operation);
                unary_specialization_t uspec = get_unary_specialization(dst_stride, dt.value_dtype().element_size(),
                                                                            src0_stride, dt.element_size());
                unary_operation_t kfunc = operation.specializations[uspec];
                if (innersize > 0) {
                    do {
                        kfunc(iter.data<0>(), dst_stride,
                                    iter.data<1>(), src0_stride,
                                    innersize, operation.auxdata);
                    } while (iter.iternext());
                }

                return DND_MOVE(result);
            }
            return ndarray_node_ref(this);
        }
        case elementwise_node_category: {
            switch (get_nop()) {
                case 1: {
                    const ndarray_node *op1 = get_opnode(0);
                    if (op1->get_category() == strided_array_node_category) {
                        ndarray_node_ref result;
                        raw_ndarray_iter<1,1> iter(m_ndim, m_shape.get(), dt.value_dtype(), result, op1);

                        intptr_t innersize = iter.innersize();
                        intptr_t dst_stride = iter.innerstride<0>();
                        intptr_t src0_stride = iter.innerstride<1>();
                        kernel_instance<unary_operation_t> operation;
                        get_unary_operation(dst_stride, src0_stride, operation);
                        if (innersize > 0) {
                            do {
                                operation.kernel(iter.data<0>(), dst_stride,
                                            iter.data<1>(), src0_stride,
                                            innersize, operation.auxdata);
                            } while (iter.iternext());
                        }

                        return DND_MOVE(result);
                    }
                    break;
                }
                case 2: {
                    const ndarray_node *op1 = get_opnode(0);
                    const ndarray_node *op2 = get_opnode(1);

                    // Special case of two strided sub-operands, requiring no intermediate buffers
                    if (op1->get_category() == strided_array_node_category &&
                                op2->get_category() == strided_array_node_category) {
                        ndarray_node_ref result;
                        raw_ndarray_iter<1,2> iter(m_ndim, m_shape.get(),
                                                    dt.value_dtype(), result,
                                                    op1, op2);
                        //iter.debug_dump(std::cout);

                        intptr_t innersize = iter.innersize();
                        intptr_t dst_stride = iter.innerstride<0>();
                        intptr_t src0_stride = iter.innerstride<1>();
                        intptr_t src1_stride = iter.innerstride<2>();
                        kernel_instance<binary_operation_t> operation;
                        get_binary_operation(dst_stride, src0_stride, src1_stride, operation);
                        if (innersize > 0) {
                            do {
                                operation.kernel(iter.data<0>(), dst_stride,
                                            iter.data<1>(), src0_stride,
                                            iter.data<2>(), src1_stride,
                                            innersize, operation.auxdata);
                            } while (iter.iternext());
                        }

                        return DND_MOVE(result);
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    debug_dump(cout, "");
    throw std::runtime_error("evaluating this expression graph is not yet supported");
}

static void print_node_category(ostream& o, ndarray_node_category cat)
{
    switch (cat) {
        case strided_array_node_category:
            o << "strided_array_node_category";
            break;
        case elementwise_node_category:
            o << "elementwise_node_category";
            break;
        case arbitrary_node_category:
            o << "arbitrary_node_category";
            break;
        default:
            o << "unknown category (" << (int)cat << ")";
            break;
    }
}

static void print_node_type(ostream& o, expr_node_type type)
{
    switch (type) {
        case strided_array_node_type:
            o << "strided_array_node_type";
            break;
        case immutable_scalar_node_type:
            o << "immutable_scalar_node_type";
            break;
        case elementwise_unary_kernel_node_type:
            o << "elementwise_unary_kernel_node_type";
            break;
        case elementwise_binary_kernel_node_type:
            o << "elementwise_binary_kernel_node_type";
            break;
        default:
            o << "unknown node type (" << (int)type << ")";
            break;
    }
}

void dnd::ndarray_node::debug_dump(ostream& o, const string& indent) const
{
    o << indent << "(\"" << node_name() << "\",\n";

    o << indent << " dtype: " << get_dtype() << "\n";
    o << indent << " ndim: " << m_ndim << "\n";
    o << indent << " shape: (";
    for (int i = 0; i < m_ndim; ++i) {
        o << m_shape[i];
        if (i != m_ndim - 1) {
            o << ", ";
        }
    }
    o << ")\n";
    o << indent << " node category: ";
    print_node_category(o, get_category());
    o << "\n";
    o << indent << " node type: ";
    print_node_type(o, m_node_type);
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

void dnd::ndarray_node::debug_dump_extra(ostream&, const string&) const
{
    // Default is no extra information
}

ndarray_node_ref dnd::apply_index_to_node(ndarray_node *node,
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

ndarray_node_ref dnd::apply_integer_index_to_node(ndarray_node *node,
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
