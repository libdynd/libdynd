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

ndarray dnd::ndarray_expr_node::evaluate() const
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
                if (op1->node_category() == strided_array_node_category &&
                            op2->node_category() == strided_array_node_category) {
                    ndarray result;
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

    throw std::runtime_error("expr_node with this many operands is not yet supported");
}

ndarray_expr_node_ptr dnd::ndarray_expr_node::apply_linear_index(
                    int new_ndim, const intptr_t *new_shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool /*allow_in_place*/)
{

    return ndarray_expr_node_ptr(
            new linear_index_expr_node(new_ndim, new_shape, axis_map,
                        index_strides, start_index, this));
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
    o << indent << " shape: ";
    for (int i = 0; i < m_ndim; ++i) {
        o << m_shape[i] << " ";
    }
    o << "\n";
    o << indent << " node category: ";
    print_node_category(o, m_node_category);
    o << "\n";
    o << indent << " node type: ";
    print_node_type(o, m_node_type);
    o << "\n";
    debug_dump_extra(o, indent);

    o << indent << " nop: " << m_nop << "\n";
    for (int i = 0; i < m_nop; ++i) {
        o << indent << " operand " << i << ":\n";
        m_opnodes[i]->debug_dump(o, indent + "  ");
    }

    o << indent << ")\n";
}

void dnd::ndarray_expr_node::debug_dump_extra(ostream&, const string&) const
{
    // Default is no extra information
}
