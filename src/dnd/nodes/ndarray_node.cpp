//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>
#include <string>

#include <dnd/ndarray.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/nodes/ndarray_node.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

#include "ndarray_expr_node_instances.hpp"

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
    switch (m_nop) {
        case 0:
            if (m_node_category == strided_array_node_category) {
                // Evaluate any expression dtype as well
                if (m_dtype.kind() == expression_kind) {
                    ndarray_node_ref result;
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

                return ndarray_node_ref(this);
            }
            break;
        case 1: {
            const ndarray_node *op1 = m_opnodes[0].get();

            if (m_node_category == elementwise_node_category) {
                if (op1->get_node_category() == strided_array_node_category) {
                    ndarray_node_ref result;
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
            const ndarray_node *op1 = m_opnodes[0].get();
            const ndarray_node *op2 = m_opnodes[1].get();

            if (m_node_category == elementwise_node_category) {
                // Special case of two strided sub-operands, requiring no intermediate buffers
                if (op1->get_node_category() == strided_array_node_category &&
                            op2->get_node_category() == strided_array_node_category) {
                    ndarray_node_ref result;
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

void dnd::ndarray_node::debug_dump(ostream& o, const string& indent) const
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

void dnd::ndarray_node::debug_dump_extra(ostream&, const string&) const
{
    // Default is no extra information
}
