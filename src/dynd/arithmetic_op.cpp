//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dynd/ndobject.hpp>
#include <dynd/ndobject_iter.hpp>
#include <dynd/dtype_promotion.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/elwise_expr_kernels.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/expr_dtype.hpp>

using namespace std;
using namespace dynd;

namespace {
    template<class OP>
    struct binary_single_kernel {
        static void func(char *dst, const char * const *src,
                        kernel_data_prefix *DYND_UNUSED(extra))
        {
            typedef typename OP::type T;
            T s0, s1, r;

            s0 = *reinterpret_cast<const T *>(src[0]);
            s1 = *reinterpret_cast<const T *>(src[1]);

            r = OP::operate(s0, s1);

            *reinterpret_cast<T *>(dst) = r;
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
            intptr_t src0_stride = src_stride[0], src1_stride = src_stride[1];

            for (size_t i = 0; i != count; ++i) {
                T s0, s1, r;
                s0 = *reinterpret_cast<const T *>(src0);
                s1 = *reinterpret_cast<const T *>(src1);

                r = OP::operate(s0, s1);

                *reinterpret_cast<T *>(dst) = r;

                dst += dst_stride;
                src0 += src0_stride;
                src1 += src1_stride;
            }
        }
    };

    struct expr_operation_pair {
        expr_single_operation_t single;
        expr_strided_operation_t strided;
    };

    class arithmetic_op_kernel_generator : public expr_kernel_generator {
        type_id_t m_tid;
        expr_operation_pair m_op_pair;
        const char *m_name;
    public:
        arithmetic_op_kernel_generator(type_id_t tid, const expr_operation_pair& op_pair,
                        const char *name)
            : expr_kernel_generator(true), m_tid(tid), m_op_pair(op_pair), m_name(name)
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
                ss << "The " << m_name << " kernel requires 2 src operands, ";
                ss << "received " << src_count;
                throw runtime_error(ss.str());
            }
            if (dst_dt.get_type_id() != m_tid || src_dt[0].get_type_id() != m_tid ||
                            src_dt[1].get_type_id() != m_tid) {
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
                    e->set_function<expr_single_operation_t>(m_op_pair.single);
                    break;
                case kernel_request_strided:
                    e->set_function<expr_strided_operation_t>(m_op_pair.strided);
                    break;
                default: {
                    stringstream ss;
                    ss << "arithmetic_op_kernel_generator: unrecognized request " << (int)kernreq;
                    throw runtime_error(ss.str());
                }
            }
            return offset_out + sizeof(kernel_data_prefix);
        }


        void print_dtype(std::ostream& o) const
        {
            o << m_name << "(op0, op1)";
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

#ifdef DYND_HAS_INT128
#define DYND_INT128_BINARY_OP_PAIR(operation) \
    {&binary_single_kernel<operation<dynd_int128> >::func, &binary_strided_kernel<operation<dynd_int128> >::func}
#else
#define DYND_INT128_BINARY_OP_PAIR(operation) {NULL, NULL}
#endif

#ifdef DYND_HAS_UINT128
#define DYND_UINT128_BINARY_OP_PAIR(operation) \
    {&binary_single_kernel<operation<dynd_uint128> >::func, &binary_strided_kernel<operation<dynd_uint128> >::func}
#else
#define DYND_UINT128_BINARY_OP_PAIR(operation) {NULL, NULL}
#endif

#ifdef DYND_HAS_FLOAT128
#define DYND_FLOAT128_BINARY_OP_PAIR(operation) \
    {&binary_single_kernel<operation<dynd_float128> >::func, &binary_strided_kernel<operation<dynd_float128> >::func}
#else
#define DYND_FLOAT128_BINARY_OP_PAIR(operation) {NULL, NULL}
#endif

#define DYND_BUILTIN_DTYPE_BINARY_OP_TABLE(operation) { \
    {&binary_single_kernel<operation<int32_t> >::func, &binary_strided_kernel<operation<int32_t> >::func}, \
    {&binary_single_kernel<operation<int64_t> >::func, &binary_strided_kernel<operation<int64_t> >::func}, \
    DYND_INT128_BINARY_OP_PAIR(operation), \
    {&binary_single_kernel<operation<int32_t> >::func, &binary_strided_kernel<operation<uint32_t> >::func}, \
    {&binary_single_kernel<operation<uint64_t> >::func, &binary_strided_kernel<operation<uint64_t> >::func}, \
    DYND_UINT128_BINARY_OP_PAIR(operation), \
    {&binary_single_kernel<operation<float> >::func, &binary_strided_kernel<operation<float> >::func}, \
    {&binary_single_kernel<operation<double> >::func, &binary_strided_kernel<operation<double> >::func}, \
    DYND_FLOAT128_BINARY_OP_PAIR(operation), \
    {&binary_single_kernel<operation<complex<float> > >::func, &binary_strided_kernel<operation<complex<float> > >::func}, \
    {&binary_single_kernel<operation<complex<double> > >::func, &binary_strided_kernel<operation<complex<double> > >::func} \
    }

#define DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(operation) \
    static const expr_operation_pair operation##_table[11] = \
                DYND_BUILTIN_DTYPE_BINARY_OP_TABLE(operation);

DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(addition);
DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(subtraction);
DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(multiplication);
DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(division);

// These operators are declared in ndobject.hpp

ndobject apply_operator(const ndobject *ops,
                const char *name,
                const expr_operation_pair *table)
{
    // Get the promoted dtype
    dtype dt = promote_dtypes_arithmetic(ops[0].get_udtype(), ops[1].get_udtype());
    static int compress_type_id[builtin_type_id_count] = {
                    -1, -1, // uninitialized, bool
                    -1, -1, 0, 1,// int8, ..., int64
                    2, // int128
                    -1, -1, 3, 4, // uint8, ..., uint64,
                    5, // uint128
                    -1, 6, 7, // float16, ..., float64
                    8, // float128
                    9, 10, // complex<float32>, complex<float64>
                    -1};
    int table_index = -1;
    if (dt.is_builtin()) {
        table_index = compress_type_id[dt.get_type_id()];
    }
    if (table_index < 0) {
        stringstream ss;
        ss << "Operator " << name << " is not supported for dynd type " << dt;
        throw runtime_error(ss.str());
    }
    const expr_operation_pair& func_ptr = table[table_index];
    if (func_ptr.single == NULL) {
        stringstream ss;
        ss << "Operator " << name << " is not supported for dynd type " << dt;
        throw runtime_error(ss.str());
    }

    // Get the broadcasted shape
    size_t undim = max(ops[0].get_undim(), ops[1].get_undim());
    dimvector result_shape(undim), tmp_shape(undim);
    for (size_t j = 0; j != undim; ++j) {
        result_shape[j] = 1;
    }
    for (size_t i = 0; i != 2; ++i) {
        size_t undim_i = ops[i].get_undim();
        if (undim_i > 0) {
            ops[i].get_shape(tmp_shape.get());
            incremental_broadcast(undim, result_shape.get(), undim_i, tmp_shape.get());
        }
    }

    // Assemble the destination value dtype
    dtype result_vdt = dt;
    for (size_t j = 0; j != undim; ++j) {
        if (result_shape[undim - j - 1] == -1) {
            result_vdt = make_var_dim_dtype(result_vdt);
        } else {
            result_vdt = make_strided_dim_dtype(result_vdt);
        }
    }

    // Create the result
    string field_names[2] = {"arg0", "arg1"};
    ndobject ops_as_dt[2] = {ops[0].ucast(dt), ops[1].ucast(dt)};
    ndobject result = combine_into_struct(2, field_names, ops_as_dt);
    // Because the expr dtype's operand is the result's dtype,
    // we can swap it in as the dtype
    dtype edt = make_expr_dtype(result_vdt,
                    result.get_dtype(),
                    new arithmetic_op_kernel_generator(dt.get_type_id(), func_ptr, name));
    edt.swap(result.get_ndo()->m_dtype);
    return result;
}

ndobject dynd::operator+(const ndobject& op1, const ndobject& op2)
{
    ndobject ops[2] = {op1, op2};
    return apply_operator(ops, "addition", addition_table);
}

ndobject dynd::operator-(const ndobject& op1, const ndobject& op2)
{
    ndobject ops[2] = {op1, op2};
    return apply_operator(ops, "subtraction", subtraction_table);
}

ndobject dynd::operator*(const ndobject& op1, const ndobject& op2)
{
    ndobject ops[2] = {op1, op2};
    return apply_operator(ops, "multiplication", multiplication_table);
}

ndobject dynd::operator/(const ndobject& op1, const ndobject& op2)
{
    ndobject ops[2] = {op1, op2};
    return apply_operator(ops, "division", division_table);
}
