//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <stdexcept>
#include <sstream>
#include <string>

#include <dnd/ndarray.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/ndarray_expr_node.hpp>
#include <dnd/raw_iteration.hpp>

#include "ndarray_expr_node_instances.hpp"

using namespace std;
using namespace dnd;

void dnd::ndarray_expr_node::as_data_and_strides(char ** /*out_data*/,
                                                intptr_t * /*out_strides*/) const
{
    throw std::runtime_error("as_data_and_strides is only valid for "
                             "nodes with an expr_node_strided_array category");
}

pair<nullary_operation_t, shared_ptr<auxiliary_data> >
                dnd::ndarray_expr_node::get_nullary_operation(intptr_t) const
{
    throw std::runtime_error("get_nullary_operation is only valid for "
                             "generator nodes which provide an implementation");
}

pair<unary_operation_t, shared_ptr<auxiliary_data> >
                dnd::ndarray_expr_node::get_unary_operation(intptr_t, intptr_t) const
{
    throw std::runtime_error("get_unary_operation is only valid for "
                             "unary nodes which provide an implementation");
}

pair<binary_operation_t, shared_ptr<auxiliary_data> >
                dnd::ndarray_expr_node::get_binary_operation(intptr_t, intptr_t, intptr_t) const
{
    throw std::runtime_error("get_binary_operation is only valid for "
                             "binary nodes which provide an implementation");
}

ndarray_expr_node_ptr dnd::ndarray_expr_node::evaluate() const
{
    switch (m_nop) {
        case 0:
            break;
        case 1:
            break;
        case 2: {
            const ndarray_expr_node *op1 = m_opnodes[0].get();
            const ndarray_expr_node *op2 = m_opnodes[1].get();

            if (m_node_category == elementwise_node_category) {
                // Special case of two strided sub-operands, requiring no intermediate buffers
                if (op1->get_node_category() == strided_array_node_category &&
                            op2->get_node_category() == strided_array_node_category) {
                    ndarray_expr_node_ptr result;
                    raw_ndarray_iter<1,2> iter(m_ndim, m_shape.get(),
                                                m_dtype, result,
                                                op1, op2);
                    //iter.debug_dump(std::cout);

                    intptr_t innersize = iter.innersize();
                    intptr_t dst_stride = iter.innerstride<0>();
                    intptr_t src0_stride = iter.innerstride<1>();
                    intptr_t src1_stride = iter.innerstride<2>();
                    pair<binary_operation_t, shared_ptr<auxiliary_data> > operation =
                                get_binary_operation(dst_stride, src0_stride, src1_stride);
                    if (innersize > 0) {
                        do {
                            operation.first(iter.data<0>(), dst_stride,
                                        iter.data<1>(), src0_stride,
                                        iter.data<2>(), src1_stride,
                                        innersize, operation.second.get());
                        } while (iter.iternext());
                    }

                    return std::move(result);
                }
            }
            break;
        }
        default:
            break;
    }

    debug_dump(cout, "");
    throw std::runtime_error("expr_node with this many operands is not yet supported");
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
        case misbehaved_strided_array_node_type:
            o << "misbehaved_strided_array_node_type";
            break;
        case convert_dtype_node_type:
            o << "convert_dtype_node_type";
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
                                char *originptr, const std::shared_ptr<void>& buffer_owner)
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
    intptr_t stride = dt.itemsize();
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

    m_buffer_owner.reset(::dnd::detail::ndarray_buffer_allocator(dt.itemsize() * num_elements),
                                ::dnd::detail::ndarray_buffer_deleter);

    m_originptr = reinterpret_cast<char *>(m_buffer_owner.get());
}

void dnd::strided_array_expr_node::as_data_and_strides(char **out_originptr,
                                                    intptr_t *out_strides) const
{
    *out_originptr = m_originptr;
    memcpy(out_strides, m_strides.get(), get_ndim() * sizeof(intptr_t));
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

// misbehaved_strided_array_expr_node

dnd::misbehaved_strided_array_expr_node::misbehaved_strided_array_expr_node(const dtype& dt, int ndim,
                                const intptr_t *shape, const intptr_t *strides,
                                char *originptr, const std::shared_ptr<void>& buffer_owner)
    : ndarray_expr_node(dt.as_nbo(), ndim, 0, shape,
                        arbitrary_node_category, misbehaved_strided_array_node_type),
      m_inner_dtype(dt), m_originptr(originptr), m_strides(ndim, strides), m_buffer_owner(buffer_owner)
{
}

void dnd::misbehaved_strided_array_expr_node::as_data_and_strides(char **out_originptr,
                                                    intptr_t *out_strides) const
{
    *out_originptr = m_originptr;
    memcpy(out_strides, m_strides.get(), get_ndim() * sizeof(intptr_t));
}

ndarray_expr_node_ptr dnd::misbehaved_strided_array_expr_node::apply_linear_index(
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
            new misbehaved_strided_array_expr_node(m_dtype, new_ndim, new_shape, new_strides.get(),
                                        new_originptr, m_buffer_owner));
    }
}

void dnd::misbehaved_strided_array_expr_node::debug_dump_extra(ostream& o, const string& indent) const
{
    o << indent << " inner dtype: " << m_inner_dtype << "\n";
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
