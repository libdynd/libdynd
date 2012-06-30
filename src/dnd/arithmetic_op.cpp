//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dnd/ndarray.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/dtype_promotion.hpp>
#include <dnd/kernels/builtin_dtype_binary_kernel_table.hpp>
#include <dnd/nodes/elementwise_binary_kernel_node.hpp>

using namespace std;
using namespace dnd;


namespace {

    class arithmetic_operator_factory {
    protected:
        dtype m_dtype;
        specialized_binary_operation_table_t *m_builtin_optable;
        const char *m_node_name;

    public:
        arithmetic_operator_factory()
            : m_dtype(), m_builtin_optable(NULL), m_node_name("")
        {
        }

        arithmetic_operator_factory(const arithmetic_operator_factory& rhs)
            : m_dtype(rhs.m_dtype), m_builtin_optable(rhs.m_builtin_optable), m_node_name(rhs.m_node_name)
        {
        }

        arithmetic_operator_factory(specialized_binary_operation_table_t *builtin_optable, const char *node_name)
            : m_dtype(), m_builtin_optable(builtin_optable), m_node_name(node_name)
        {
        }

        arithmetic_operator_factory& operator=(arithmetic_operator_factory& rhs)
        {
            m_dtype = rhs.m_dtype;
            m_builtin_optable = rhs.m_builtin_optable;
            m_node_name = rhs.m_node_name;

            return *this;
        }

        void promote_dtypes(const dtype& dt1, const dtype& dt2) {
            //cout << "promoting dtypes " << dt1 << ", " << dt2 << endl;
            m_dtype = promote_dtypes_arithmetic(dt1, dt2);
            //cout << "to " << m_dtype << endl;
        }

        const dtype& get_dtype(int) {
            return m_dtype;
        }

        void swap(arithmetic_operator_factory& that) {
            m_dtype.swap(that.m_dtype);
            std::swap(m_builtin_optable, that.m_builtin_optable);
            std::swap(m_node_name, that.m_node_name);
        }

        void get_binary_operation(intptr_t dst_fixedstride, intptr_t src1_fixedstride,
                                    intptr_t src2_fixedstride,
                                    kernel_instance<binary_operation_t>& out_kernel) const
        {
            out_kernel.kernel = get_binary_operation_from_builtin_dtype_table(m_builtin_optable,
                                m_dtype, dst_fixedstride,
                                src1_fixedstride, src2_fixedstride);
            out_kernel.auxdata.free();
        }

        const char *node_name() const {
            return m_node_name;
        }
    };

} // anonymous namespace

/*
static ndarray arithmetic_op(const ndarray& op0, const ndarray& op1,
                                            binary_operation_t builtin_optable[][6])
{
    dtype dt = promote_dtypes_arithmetic(op0.get_dtype(), op1.get_dtype());
    ndarray result;
    raw_ndarray_iter<1,2> iter(dt, result, op0, op1);
    //cout << "src0:\n" << op0 << "\n";
    //op0.debug_dump(cout);
    //cout << "\n";
    //cout << "src1:\n" << op1 << "\n";
    //op1.debug_dump(cout);
    //cout << "\n";
    //cout << "dst:\n";
    //result.debug_dump(cout);
    //cout << "\n";

    if (dt.extended() != NULL) {
        throw std::runtime_error("arithmetic operations for extended dtypes isn't implemented yet");
    }

    // Use buffering if a dtype conversion or alignment operation
    // is required.
    if (dt != op0.get_dtype() || dt != op1.get_dtype() ||
                !dt.is_data_aligned(iter.get_align_test<1>()) ||
                !dt.is_data_aligned(iter.get_align_test<2>())) {
        stringstream ss;
        ss << "arithmetic operation buffering isn't implemented yet, dtypes ";
        ss << op0.get_dtype() << " " << op1.get_dtype();
        throw std::runtime_error(ss.str());
    } else {
        intptr_t innersize = iter.innersize();
        intptr_t dst_stride = iter.innerstride<0>();
        intptr_t src0_stride = iter.innerstride<1>();
        intptr_t src1_stride = iter.innerstride<2>();
        binary_operation_t operation = get_builtin_operation_function(builtin_optable,
                                            dt, dst_stride, src0_stride, src1_stride);
        if (innersize > 0) {
            do {
                operation(iter.data<0>(), dst_stride,
                            iter.data<1>(), src0_stride,
                            iter.data<2>(), src1_stride,
                            innersize, NULL);
            } while (iter.iternext());
        }
    }

    return DND_MOVE(result);
}
*/

namespace {
    template<class T>
    struct addition {
        typedef T type;
        static inline T operate(T x, T y) {
            return x + y;
        }
    };

    template<class T>
    struct subtraction {
        typedef T type;
        static inline T operate(T x, T y) {
            return x - y;
        }
    };

    template<class T>
    struct multiplication {
        typedef T type;
        static inline T operate(T x, T y) {
            return x * y;
        }
    };

    template<class T>
    struct division {
        typedef T type;
        static inline T operate(T x, T y) {
            return x / y;
        }
    };


} // anonymous namespace


static DND_BUILTIN_DTYPE_BINARY_OPERATION_TABLE(addition);
static DND_BUILTIN_DTYPE_BINARY_OPERATION_TABLE(subtraction);
static DND_BUILTIN_DTYPE_BINARY_OPERATION_TABLE(multiplication);
static DND_BUILTIN_DTYPE_BINARY_OPERATION_TABLE(division);

// These operators are declared in ndarray.hpp

ndarray dnd::operator+(const ndarray& op1, const ndarray& op2)
{
    arithmetic_operator_factory op_factory(builtin_addition_table, "add");
    return ndarray(make_elementwise_binary_kernel_node(op1.get_expr_tree(), op2.get_expr_tree(), op_factory,
                                    assign_error_fractional));
}

ndarray dnd::operator-(const ndarray& op1, const ndarray& op2)
{
    arithmetic_operator_factory op_factory(builtin_subtraction_table, "subtract");
    return ndarray(make_elementwise_binary_kernel_node(op1.get_expr_tree(), op2.get_expr_tree(), op_factory,
                                    assign_error_fractional));
}

ndarray dnd::operator*(const ndarray& op1, const ndarray& op2)
{
    arithmetic_operator_factory op_factory(builtin_multiplication_table, "multiply");
    return ndarray(make_elementwise_binary_kernel_node(op1.get_expr_tree(), op2.get_expr_tree(), op_factory,
                                    assign_error_fractional));
}

ndarray dnd::operator/(const ndarray& op1, const ndarray& op2)
{
    arithmetic_operator_factory op_factory(builtin_division_table, "divide");
    return ndarray(make_elementwise_binary_kernel_node(op1.get_expr_tree(), op2.get_expr_tree(), op_factory,
                                    assign_error_fractional));
}
