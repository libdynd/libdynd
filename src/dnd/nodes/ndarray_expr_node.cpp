//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <string>

#include <dnd/ndarray.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/nodes/ndarray_expr_node.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

#include "ndarray_expr_node_instances.hpp"

using namespace std;
using namespace dnd;

void dnd::ndarray_expr_node::as_readwrite_data_and_strides(int ndim, char ** DND_UNUSED(out_data),
                                                intptr_t * DND_UNUSED(out_strides)) const
{
    throw std::runtime_error("as_readwrite_data_and_strides is only valid for "
                             "nodes with an expr_node_strided_array category");
}

void dnd::ndarray_expr_node::as_readonly_data_and_strides(int ndim, char const ** DND_UNUSED(out_data),
                                                intptr_t * DND_UNUSED(out_strides)) const
{
    throw std::runtime_error("as_readonly_data_and_strides is only valid for "
                             "nodes with an expr_node_strided_array category");
}

void dnd::ndarray_expr_node::get_nullary_operation(intptr_t, kernel_instance<nullary_operation_t>&) const
{
    throw std::runtime_error("get_nullary_operation is only valid for "
                             "generator nodes which provide an implementation");
}

void dnd::ndarray_expr_node::get_unary_operation(intptr_t, intptr_t, kernel_instance<unary_operation_t>&) const
{
    throw std::runtime_error("get_unary_operation is only valid for "
                             "unary nodes which provide an implementation");
}

void dnd::ndarray_expr_node::get_binary_operation(intptr_t, intptr_t, intptr_t, kernel_instance<binary_operation_t>&) const
{
    throw std::runtime_error("get_binary_operation is only valid for "
                             "binary nodes which provide an implementation");
}

ndarray_expr_node_ptr dnd::ndarray_expr_node::evaluate()
{
    switch (m_nop) {
        case 0:
            if (m_node_category == strided_array_node_category) {
                // Evaluate any expression dtype as well
                if (m_dtype.kind() == expression_kind) {
                    ndarray_expr_node_ptr result;
                    raw_ndarray_iter<1,1> iter(m_ndim, m_shape.get(), m_dtype.value_dtype(), result, this);

                    intptr_t innersize = iter.innersize();
                    intptr_t dst_stride = iter.innerstride<0>();
                    intptr_t src0_stride = iter.innerstride<1>();
                    unary_specialization_kernel_instance operation;
                    get_dtype_assignment_kernel(m_dtype.value_dtype(), m_dtype, assign_error_none, operation);
                    unary_specialization_t uspec = get_unary_specialization(dst_stride, m_dtype.value_dtype().element_size(),
                                                                                src0_stride, m_dtype.element_size());
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

                return ndarray_expr_node_ptr(this);
            }
            break;
        case 1: {
            const ndarray_expr_node *op1 = m_opnodes[0].get();

            if (m_node_category == elementwise_node_category) {
                if (op1->get_node_category() == strided_array_node_category) {
                    ndarray_expr_node_ptr result;
                    raw_ndarray_iter<1,1> iter(m_ndim, m_shape.get(), m_dtype.value_dtype(), result, op1);

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
            }
            break;
        }
        case 2: {
            const ndarray_expr_node *op1 = m_opnodes[0].get();
            const ndarray_expr_node *op2 = m_opnodes[1].get();

            if (m_node_category == elementwise_node_category) {
                // Special case of two strided sub-operands, requiring no intermediate buffers
                if (op1->get_node_category() == strided_array_node_category &&
                            op2->get_node_category() == strided_array_node_category) {
                    ndarray_expr_node_ptr result;
                    raw_ndarray_iter<1,2> iter(m_ndim, m_shape.get(),
                                                m_dtype.value_dtype(), result,
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
            }
            break;
        }
        default:
            break;
    }

    debug_dump(cout, "");
    throw std::runtime_error("evaluating this expression graph is not yet supported");
}

static void print_node_category(ostream& o, expr_node_category cat)
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
        case broadcast_shape_node_type:
            o << "broadcast_shape_node_type";
            break;
        case elementwise_binary_op_node_type:
            o << "elementwise_binary_op_node_type";
            break;
        case linear_index_node_type:
            o << "linear_index_node_type";
            break;
        default:
            o << "unknown type (" << (int)type << ")";
            break;
    }
}

void dnd::ndarray_expr_node::debug_dump(ostream& o, const string& indent) const
{
    o << indent << "(\"" << node_name() << "\",\n";

    o << indent << " dtype: " << m_dtype << "\n";
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
    print_node_category(o, m_node_category);
    o << "\n";
    o << indent << " node type: ";
    print_node_type(o, m_node_type);
    o << "\n";
    debug_dump_extra(o, indent);

    if (m_nop > 0) {
        o << indent << " nop: " << m_nop << "\n";
        for (int i = 0; i < m_nop; ++i) {
            o << indent << " operand " << i << ":\n";
            m_opnodes[i]->debug_dump(o, indent + "  ");
        }
    }

    o << indent << ")\n";
}

void dnd::ndarray_expr_node::debug_dump_extra(ostream&, const string&) const
{
    // Default is no extra information
}

// strided_array_expr_node

dnd::strided_array_expr_node::strided_array_expr_node(const dtype& dt, int ndim,
                                const intptr_t *shape, const intptr_t *strides,
                                char *originptr, const memory_block_ref& memblock)
    : ndarray_expr_node(dt, ndim, 0, shape, strided_array_node_category, strided_array_node_type),
      m_originptr(originptr), m_strides(ndim, strides), m_memblock(memblock)
{
}

dnd::strided_array_expr_node::strided_array_expr_node(const dtype& dt, int ndim,
                                const intptr_t *shape, const int *axis_perm)
    : ndarray_expr_node(dt, ndim, 0, shape, strided_array_node_category, strided_array_node_type),
      m_originptr(NULL), m_strides(ndim), m_memblock()
{
    // Build the strides using the ordering and shape
    intptr_t num_elements = 1;
    intptr_t stride = dt.element_size();
    for (int i = 0; i < ndim; ++i) {
        int p = axis_perm[i];
        intptr_t size = shape[p];
        if (size == 1) {
            m_strides[p] = 0;
        } else {
            m_strides[p] = stride;
            stride *= size;
            num_elements *= size;
        }
    }

    m_memblock = make_fixed_size_pod_memory_block(dt.alignment(), dt.element_size() * num_elements, &m_originptr);
}

void dnd::strided_array_expr_node::as_readwrite_data_and_strides(int ndim, char **out_originptr,
                                                    intptr_t *out_strides) const
{
    *out_originptr = m_originptr;
    if (ndim == m_ndim) {
        memcpy(out_strides, m_strides.get(), m_ndim * sizeof(intptr_t));
    } else {
        memset(out_strides, 0, (ndim - m_ndim) * sizeof(intptr_t));
        memcpy(out_strides + (ndim - m_ndim), m_strides.get(), m_ndim * sizeof(intptr_t));
    }
}

void dnd::strided_array_expr_node::as_readonly_data_and_strides(int ndim, char const **out_originptr,
                                                    intptr_t *out_strides) const
{
    *out_originptr = m_originptr;
    if (ndim == m_ndim) {
        memcpy(out_strides, m_strides.get(), m_ndim * sizeof(intptr_t));
    } else {
        memset(out_strides, 0, (ndim - m_ndim) * sizeof(intptr_t));
        memcpy(out_strides + (ndim - m_ndim), m_strides.get(), m_ndim * sizeof(intptr_t));
    }
}

ndarray_expr_node_ptr dnd::strided_array_expr_node::as_dtype(const dtype& dt,
                    dnd::assign_error_mode errmode, bool allow_in_place)
{
    if (allow_in_place) {
        m_dtype = make_conversion_dtype(dt, m_dtype, errmode);
        return ndarray_expr_node_ptr(this);
    } else {
        return ndarray_expr_node_ptr(new strided_array_expr_node(
                        make_conversion_dtype(dt, m_dtype, errmode),
                        m_ndim, m_shape.get(), m_strides.get(), m_originptr, m_memblock));
    }
}

ndarray_expr_node_ptr dnd::strided_array_expr_node::apply_linear_index(
                int ndim, const bool *remove_axis,
                const intptr_t *start_index, const intptr_t *index_strides,
                const intptr_t *shape,
                bool allow_in_place)
{
    /*
    cout << "Applying linear index:\n";
    cout << "ndim: " << ndim << "\n";
    cout << "remove_axis: ";
    for (int i = 0; i < ndim; ++i) {
        cout << remove_axis[i] << " ";
    }
    cout << "\n";
    cout << "start_index: ";
    for (int i = 0; i < ndim; ++i) {
        cout << start_index[i] << " ";
    }
    cout << "\n";
    cout << "index_strides: ";
    for (int i = 0; i < ndim; ++i) {
        cout << index_strides[i] << " ";
    }
    cout << "\n";
    cout << "shape: ";
    for (int i = 0; i < ndim; ++i) {
        cout << shape[i] << " ";
    }
    cout << "\n";
    */

    // Ignore the leftmost dimensions to which this node would broadcast
    if (ndim > m_ndim) {
        remove_axis += (ndim - m_ndim);
        start_index += (ndim - m_ndim);
        index_strides += (ndim - m_ndim);
        shape += (ndim - m_ndim);
        ndim = m_ndim;
    }

    // For each axis not being removed, apply the start_index and index_strides
    // to originptr and the node's strides, respectively. At the same time,
    // apply the remove_axis compression to the strides and shape.
    if (allow_in_place) {
        int j = 0;
        for (int i = 0; i < m_ndim; ++i) {
            m_originptr += m_strides[i] * start_index[i];
            if (!remove_axis[i]) {
                if (m_shape[i] != 1) {
                    m_strides[j] = m_strides[i] * index_strides[i];
                    m_shape[j] = shape[i];
                } else {
                    m_strides[j] = 0;
                    m_shape[j] = 1;
                }
                ++j;
            }
        }
        m_ndim = j;

        return ndarray_expr_node_ptr(this);
    } else {
        // Apply the start_index to m_originptr
        char *new_originptr = m_originptr;
        dimvector new_strides(m_ndim);
        dimvector new_shape(m_ndim);

        int j = 0;
        for (int i = 0; i < m_ndim; ++i) {
            new_originptr += m_strides[i] * start_index[i];
            if (!remove_axis[i]) {
                if (m_shape[i] != 1) {
                    new_strides[j] = m_strides[i] * index_strides[i];
                    new_shape[j] = shape[i];
                } else {
                    new_strides[j] = 0;
                    new_shape[j] = 1;
                }
                ++j;
            }
        }
        ndim = j;

        return ndarray_expr_node_ptr(
            new strided_array_expr_node(m_dtype, ndim, new_shape.get(), new_strides.get(),
                                        new_originptr, m_memblock));
    }
}

void dnd::strided_array_expr_node::debug_dump_extra(ostream& o, const string& indent) const
{
    o << indent << " strides: (";
    for (int i = 0; i < m_ndim; ++i) {
        o << m_strides[i];
        if (i != m_ndim - 1) {
            o << ", ";
        }
    }
    o << ")\n";
    o << indent << " originptr: " << (void *)m_originptr << "\n";
    o << indent << " memoryblock owning the data:\n";
    memory_block_debug_dump(m_memblock.get(), o, indent + " ");
}
