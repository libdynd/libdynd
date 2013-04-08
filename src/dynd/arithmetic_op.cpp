//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dynd/ndobject.hpp>
#include <dynd/ndobject_iter.hpp>
#include <dynd/dtype_promotion.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

using namespace std;
using namespace dynd;

namespace {
    template<class OP>
    struct binary_single_kernel {
        static void func(char *dst, const char **src,
                        kernel_data_prefix *DYND_UNUSED(extra))
        {
            typedef typename OP::type T;

            *reinterpret_cast<T *>(dst) = OP::operate(
                                        *reinterpret_cast<const T *>(src[0]),
                                        *reinterpret_cast<const T *>(src[1]));
        }
    };

    template<class OP>
    struct binary_strided_kernel {
        static void func(char *dst, intptr_t dst_stride,
                        const char * const *src, const intptr_t *src_stride,
                        size_t count, kernel_data_prefix *DYND_UNUSED(extra))
        {
            typedef typename OP::type T;
            const char *src0 = src[0], *src1 = src[1];
            intptr_t src_stride0 = src_stride[0], src_stride1 = src_stride[1];

            for (intptr_t i = 0; i < count; ++i) {
                *reinterpret_cast<T *>(dst) = OP::operate(
                                            *reinterpret_cast<const T *>(src0),
                                            *reinterpret_cast<const T *>(src1));
                dst += dst_stride;
                src0 += src0_stride;
                src1 += src1_stride;
            }
        }
    };

    template<class OP>
    class arithmetic_op_kernel_generator : public expr_kernel_generator {
    public:
        arithmetic_op_kernel_generator()
            : expr_kernel_generator(true)
        {
        }

        virtual ~arithmetic_op_kernel_generator() {
        }

        size_t make_expr_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const dtype& dst_dt, const char *dst_metadata,
                    size_t src_count, const dtype *src_dt, const char **src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const
        {
            if (src_count != 2) {
                stringstream ss;
                ss << "The " << OP::name() << " kernel requires 2 src operands, ";
                ss << "received " << src_count;
                throw runtime_error(ss.str());
            }
            type_id_t tid = type_id_of<OP::type>::value;
            if (dst_dt.get_type_id() != tid || src_dt[0].get_type_id() != tid ||
                            src_dt[1].get_type_id() != tid) {
                // If the dtypes don't match the ones for this generator,
                // call the elementwise dimension handler to handle one dimension
                // or handle input/output buffering, giving 'this' as the next
                // kernel generator to call
                return make_elwise_dimension_expr_kernel(out, offset_out,
                                dst_dt, dst_metadata,
                                src_count, src_dt, src_metadata,
                                kernreq, ectx,
                                this);
            }
            // This is a leaf kernel, so no additional allocation is needed
            kernel_data_prefix *e = out->get_at<kernel_data_prefix>(offset_out);
            switch (kernreq) {
                case kernel_request_single:
                    e->set_function<expr_single_operation_t>(&binary_single_kernel<OP>::func);
                    break;
                case kernel_request_strided:
                    e->set_function<expr_strided_operation_t>(&binary_strided_kernel<OP>::func);
                    break;
                default: {
                    stringstream ss;
                    ss << "arithmetic_op_kernel_generator: unrecognized request " << (int)kernreq;
                    throw runtime_error(ss.str());
                }
            }
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
        inline static const char *name() {return "addition";}
    };

    template<class T>
    struct subtraction {
        typedef T type;
        static inline T operate(T x, T y) {
            return x - y;
        }
        inline static const char *name() {return "subtraction";}
    };

    template<class T>
    struct multiplication {
        typedef T type;
        static inline T operate(T x, T y) {
            return x * y;
        }
        inline static const char *name() {return "multiplication";}
    };

    template<class T>
    struct division {
        typedef T type;
        static inline T operate(T x, T y) {
            return x / y;
        }
        inline static const char *name() {return "division";}
    };
} // anonymous namespace

#if 0
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
#endif