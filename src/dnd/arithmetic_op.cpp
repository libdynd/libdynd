//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dnd/ndarray.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/dtype_promotion.hpp>
#include <dnd/kernels/builtin_dtype_binary_kernel_table.hpp>
#include <dnd/nodes/elwise_binary_kernel_node.hpp>

using namespace std;
using namespace dynd;


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
                                    const eval::eval_context *DND_UNUSED(ectx),
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

ndarray dynd::operator+(const ndarray& op1, const ndarray& op2)
{
    dtype dt = promote_dtypes_arithmetic(op1.get_dtype(), op2.get_dtype());
    kernel_instance<binary_operation_t> kernel;
    // TODO: This is throwing away the stride specialization, probably want to put that back
    //       in how it is with the unary_specialization instance
    kernel.kernel = get_binary_operation_from_builtin_dtype_table(builtin_addition_table, dt,
                    numeric_limits<intptr_t>::max(),
                    numeric_limits<intptr_t>::max(),
                    numeric_limits<intptr_t>::max());
    return ndarray(make_elwise_binary_kernel_node_steal_kernel(dt,
                    op1.get_node()->as_dtype(dt), op2.get_node()->as_dtype(dt), kernel));
}

ndarray dynd::operator-(const ndarray& op1, const ndarray& op2)
{
    dtype dt = promote_dtypes_arithmetic(op1.get_dtype(), op2.get_dtype());
    kernel_instance<binary_operation_t> kernel;
    // TODO: This is throwing away the stride specialization, probably want to put that back
    //       in how it is with the unary_specialization instance
    kernel.kernel = get_binary_operation_from_builtin_dtype_table(builtin_subtraction_table, dt,
                    numeric_limits<intptr_t>::max(),
                    numeric_limits<intptr_t>::max(),
                    numeric_limits<intptr_t>::max());
    return ndarray(make_elwise_binary_kernel_node_steal_kernel(dt,
                    op1.get_node()->as_dtype(dt), op2.get_node()->as_dtype(dt), kernel));
}

ndarray dynd::operator*(const ndarray& op1, const ndarray& op2)
{
    dtype dt = promote_dtypes_arithmetic(op1.get_dtype(), op2.get_dtype());
    kernel_instance<binary_operation_t> kernel;
    // TODO: This is throwing away the stride specialization, probably want to put that back
    //       in how it is with the unary_specialization instance
    kernel.kernel = get_binary_operation_from_builtin_dtype_table(builtin_multiplication_table, dt,
                    numeric_limits<intptr_t>::max(),
                    numeric_limits<intptr_t>::max(),
                    numeric_limits<intptr_t>::max());
    return ndarray(make_elwise_binary_kernel_node_steal_kernel(dt,
                    op1.get_node()->as_dtype(dt), op2.get_node()->as_dtype(dt), kernel));
}

ndarray dynd::operator/(const ndarray& op1, const ndarray& op2)
{
    dtype dt = promote_dtypes_arithmetic(op1.get_dtype(), op2.get_dtype());
    kernel_instance<binary_operation_t> kernel;
    // TODO: This is throwing away the stride specialization, probably want to put that back
    //       in how it is with the unary_specialization instance
    kernel.kernel = get_binary_operation_from_builtin_dtype_table(builtin_division_table, dt,
                    numeric_limits<intptr_t>::max(),
                    numeric_limits<intptr_t>::max(),
                    numeric_limits<intptr_t>::max());
    return ndarray(make_elwise_binary_kernel_node_steal_kernel(dt,
                    op1.get_node()->as_dtype(dt), op2.get_node()->as_dtype(dt), kernel));
}
