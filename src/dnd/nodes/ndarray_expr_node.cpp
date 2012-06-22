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

void dnd::ndarray_expr_node::as_readwrite_data_and_strides(char ** DND_UNUSED(out_data),
                                                intptr_t * DND_UNUSED(out_strides)) const
{
    throw std::runtime_error("as_readwrite_data_and_strides is only valid for "
                             "nodes with an expr_node_strided_array category");
}

void dnd::ndarray_expr_node::as_readonly_data_and_strides(char const ** DND_UNUSED(out_data),
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

ndarray_expr_node_ptr dnd::ndarray_expr_node::apply_linear_index(
                    int new_ndim, const intptr_t *new_shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool /*allow_in_place*/)
{
    /*
    cout << "Applying linear index changing " << m_ndim << " to " << new_ndim << "\n";
    cout << "New shape: ";
    for (int i = 0; i < new_ndim; ++i) cout << new_shape[i] << " ";
    cout << "\n";
    cout << "Axis map: ";
    for (int i = 0; i < new_ndim; ++i) cout << axis_map[i] << " ";
    cout << "\n";
    cout << "Index strides: ";
    for (int i = 0; i < new_ndim; ++i) cout << index_strides[i] << " ";
    cout << "\n";
    cout << "Start index: ";
    for (int i = 0; i < m_ndim; ++i) cout << start_index[i] << " ";
    cout << "\n";
    */

    return ndarray_expr_node_ptr(
            new linear_index_expr_node(new_ndim, new_shape, axis_map,
                        index_strides, start_index, this));
}

ndarray_expr_node_ptr dnd::ndarray_expr_node::apply_integer_index(int axis, intptr_t idx,
                                                    bool allow_in_place)
{
    // TODO: Create a specific integer_index_expr_node
    //
    // For now, convert to a linear index.
    int new_ndim = m_ndim - 1;
    shortvector<intptr_t> new_shape(new_ndim);
    shortvector<int> axis_map(new_ndim);
    shortvector<intptr_t> index_strides(new_ndim);
    shortvector<intptr_t> start_index(m_ndim);

    start_index[axis] = idx;
    for (int i = 0; i < new_ndim; ++i) {
        int old_i = (i < axis) ? i : i + 1;
        new_shape[i] = m_shape[old_i];
        axis_map[i] = old_i;
        index_strides[i] = 1;
        start_index[old_i] = 0;
    }

    return apply_linear_index(new_ndim, new_shape.get(), axis_map.get(), index_strides.get(),
                            start_index.get(), allow_in_place);
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
                                char *originptr, const dnd::shared_ptr<void>& buffer_owner)
    : ndarray_expr_node(dt, ndim, 0, shape, strided_array_node_category, strided_array_node_type),
      m_originptr(originptr), m_strides(ndim, strides), m_buffer_owner(buffer_owner)
{
}

dnd::strided_array_expr_node::strided_array_expr_node(const dtype& dt, int ndim,
                                const intptr_t *shape, const int *axis_perm)
    : ndarray_expr_node(dt, ndim, 0, shape, strided_array_node_category, strided_array_node_type),
      m_originptr(NULL), m_strides(ndim), m_buffer_owner()
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

    m_buffer_owner.reset(::dnd::detail::ndarray_buffer_allocator(dt.element_size() * num_elements),
                                ::dnd::detail::ndarray_buffer_deleter);

    m_originptr = reinterpret_cast<char *>(m_buffer_owner.get());
}

void dnd::strided_array_expr_node::as_readwrite_data_and_strides(char **out_originptr,
                                                    intptr_t *out_strides) const
{
    *out_originptr = m_originptr;
    memcpy(out_strides, m_strides.get(), get_ndim() * sizeof(intptr_t));
}

void dnd::strided_array_expr_node::as_readonly_data_and_strides(char const **out_originptr,
                                                    intptr_t *out_strides) const
{
    *out_originptr = m_originptr;
    memcpy(out_strides, m_strides.get(), get_ndim() * sizeof(intptr_t));
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
                        m_ndim, m_shape.get(), m_strides.get(), m_originptr, m_buffer_owner));
    }
}

ndarray_expr_node_ptr dnd::strided_array_expr_node::broadcast_to_shape(int ndim,
                        const intptr_t *shape, bool allow_in_place)
{
    intptr_t *orig_shape = m_shape.get(), *orig_strides = m_strides.get();

    // If the shape is identical, don't make a new node
    if (ndim == m_ndim && memcmp(orig_shape, shape, ndim * sizeof(intptr_t)) == 0) {
        return ndarray_expr_node_ptr(this);
    }

    if (allow_in_place) {
        // NOTE: It is required that all strides == 0 where the shape == 1

        if (ndim == m_ndim) {
            memcpy(orig_shape, shape, ndim * sizeof(intptr_t));
        } else {
            // Create the broadcast shape/strides
            dimvector newshape(ndim), newstrides(ndim);
            memcpy(newshape.get(), shape, ndim * sizeof(intptr_t));
            for (int i = 0; i < ndim - m_ndim; ++i) {
                newstrides[i] = 0;
            }
            memcpy(newstrides.get() + ndim - m_ndim, orig_strides, m_ndim * sizeof(intptr_t));

            // swap them in place
            m_ndim = ndim;
            newshape.swap(m_shape);
            newstrides.swap(m_strides);
        }
        return ndarray_expr_node_ptr(this);
    } else {
            // Create the broadcast shape/strides
            dimvector newstrides(ndim);
            for (int i = 0; i < ndim - m_ndim; ++i) {
                newstrides[i] = 0;
            }
            memcpy(newstrides.get() + ndim - m_ndim, orig_strides, m_ndim * sizeof(intptr_t));

            return ndarray_expr_node_ptr(new strided_array_expr_node(m_dtype, ndim, shape, newstrides.get(), m_originptr, m_buffer_owner));
    }
}




ndarray_expr_node_ptr dnd::strided_array_expr_node::apply_linear_index(
                    int new_ndim, const intptr_t *new_shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place)
{
    if (allow_in_place) {
        // Apply the start_index to m_originptr
        for (int i = 0; i < m_ndim; ++i) {
            m_originptr += m_strides[i] * start_index[i];
        }

        // Adopt the new shape
        m_ndim = new_ndim;
        memcpy(m_shape.get(), new_shape, new_ndim * sizeof(intptr_t));

        // Construct the new strides
        dimvector new_strides(m_ndim);
        for (int i = 0; i < new_ndim; ++i) {
            new_strides[i] = m_strides[axis_map[i]] * index_strides[i];
        }
        m_strides.swap(new_strides);

        return ndarray_expr_node_ptr(this);
    } else {
        // Apply the start_index to m_originptr
        char *new_originptr = m_originptr;
        for (int i = 0; i < m_ndim; ++i) {
            new_originptr += m_strides[i] * start_index[i];
        }

        // Construct the new strides
        dimvector new_strides(m_ndim);
        for (int i = 0; i < new_ndim; ++i) {
            new_strides[i] = m_strides[axis_map[i]] * index_strides[i];
        }

        return ndarray_expr_node_ptr(
            new strided_array_expr_node(m_dtype, new_ndim, new_shape, new_strides.get(),
                                        new_originptr, m_buffer_owner));
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
    o << indent << " buffer owner: " << m_buffer_owner.get() << "\n";
}
