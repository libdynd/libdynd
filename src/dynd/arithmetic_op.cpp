//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dynd/array.hpp>
#include <dynd/array_iter.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/elwise_expr_kernels.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/expr_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/kernels/string_algorithm_kernels.hpp>

using namespace std;
using namespace dynd;

namespace {
    template<class OP>
    struct binary_single_kernel {
        static void func(char *dst, const char * const *src,
                        ckernel_prefix *DYND_UNUSED(extra))
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
                        size_t count, ckernel_prefix *DYND_UNUSED(extra))
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

    template<class extra_type>
    class arithmetic_op_kernel_generator : public expr_kernel_generator {
        ndt::type m_rdt, m_op1dt, m_op2dt;
        expr_operation_pair m_op_pair;
        const char *m_name;
    public:
        arithmetic_op_kernel_generator(const ndt::type& rdt, const ndt::type& op1dt, const ndt::type& op2dt,
                        const expr_operation_pair& op_pair, const char *name)
            : expr_kernel_generator(true), m_rdt(rdt), m_op1dt(op1dt), m_op2dt(op2dt),
                            m_op_pair(op_pair), m_name(name)
        {
        }

        virtual ~arithmetic_op_kernel_generator() {
        }

        size_t make_expr_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const ndt::type& dst_tp, const char *dst_metadata,
                    size_t src_count, const ndt::type *src_tp, const char **src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const
        {
            if (src_count != 2) {
                stringstream ss;
                ss << "The " << m_name << " kernel requires 2 src operands, ";
                ss << "received " << src_count;
                throw runtime_error(ss.str());
            }
            if (dst_tp != m_rdt || src_tp[0] != m_op1dt ||
                            src_tp[1] != m_op2dt) {
                // If the types don't match the ones for this generator,
                // call the elementwise dimension handler to handle one dimension
                // or handle input/output buffering, giving 'this' as the next
                // kernel generator to call
                return make_elwise_dimension_expr_kernel(out, offset_out,
                                dst_tp, dst_metadata,
                                src_count, src_tp, src_metadata,
                                kernreq, ectx,
                                this);
            }
            // This is a leaf kernel, so no additional allocation is needed
            extra_type *e = out->get_at<extra_type>(offset_out);
            switch (kernreq) {
                case kernel_request_single:
                    e->base().set_function(m_op_pair.single);
                    break;
                case kernel_request_strided:
                    e->base().set_function(m_op_pair.strided);
                    break;
                default: {
                    stringstream ss;
                    ss << "generic_kernel_generator: unrecognized request " << (int)kernreq;
                    throw runtime_error(ss.str());
                }
            }
            e->init(2, dst_metadata, src_metadata);
            return offset_out + sizeof(extra_type);
        }


        void print_type(std::ostream& o) const
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
    {&binary_single_kernel<operation<dynd_complex<float> > >::func, &binary_strided_kernel<operation<dynd_complex<float> > >::func}, \
    {&binary_single_kernel<operation<dynd_complex<double> > >::func, &binary_strided_kernel<operation<dynd_complex<double> > >::func} \
    }

#define DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(operation) \
    static const expr_operation_pair operation##_table[11] = \
                DYND_BUILTIN_DTYPE_BINARY_OP_TABLE(operation);

DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(addition);
DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(subtraction);
DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(multiplication);
DYND_BUILTIN_DTYPE_BINARY_OP_TABLE_DEFS(division);

// These operators are declared in nd::array.hpp

// Get the table index by compressing the type_id's we do implement
static int compress_builtin_type_id[builtin_type_id_count] = {
                -1, -1, // uninitialized, bool
                -1, -1, 0, 1,// int8, ..., int64
                2, // int128
                -1, -1, 3, 4, // uint8, ..., uint64,
                5, // uint128
                -1, 6, 7, // float16, ..., float64
                8, // float128
                9, 10, // complex<float32>, complex<float64>
                -1};

template<class KD>
nd::array apply_binary_operator(const nd::array *ops,
                const ndt::type& rdt, const ndt::type& op1dt, const ndt::type& op2dt,
                expr_operation_pair expr_ops,
                const char *name)
{
    if (expr_ops.single == NULL) {
        stringstream ss;
        ss << "Operator " << name << " is not supported for dynd types ";
        ss << op1dt << " and " << op2dt;
        throw runtime_error(ss.str());
    }

    // Get the broadcasted shape
    size_t ndim = max(ops[0].get_ndim(), ops[1].get_ndim());
    dimvector result_shape(ndim), tmp_shape(ndim);
    for (size_t j = 0; j != ndim; ++j) {
        result_shape[j] = 1;
    }
    for (size_t i = 0; i != 2; ++i) {
        size_t ndim_i = ops[i].get_ndim();
        if (ndim_i > 0) {
            ops[i].get_shape(tmp_shape.get());
            incremental_broadcast(ndim, result_shape.get(), ndim_i, tmp_shape.get());
        }
    }

    // Assemble the destination value type
    ndt::type result_vdt = ndt::make_type(ndim, result_shape.get(), rdt);

    // Create the result
    string field_names[2] = {"arg0", "arg1"};
    nd::array ops_as_dt[2] = {ops[0].ucast(op1dt), ops[1].ucast(op2dt)};
    nd::array result = combine_into_struct(2, field_names, ops_as_dt);
    // Because the expr type's operand is the result's type,
    // we can swap it in as the type
    ndt::type edt = ndt::make_expr(result_vdt,
                    result.get_type(),
                    new arithmetic_op_kernel_generator<KD>(rdt, op1dt, op2dt, expr_ops, name));
    edt.swap(result.get_ndo()->m_type);
    return result;
}

nd::array nd::operator+(const nd::array& op1, const nd::array& op2)
{
    nd::array ops[2] = {op1, op2};
    expr_operation_pair func_ptr;
    ndt::type op1dt = op1.get_dtype().value_type();
    ndt::type op2dt = op2.get_dtype().value_type();
    if (op1dt.is_builtin() && op1dt.is_builtin()) {
        ndt::type rdt = promote_types_arithmetic(op1dt, op2dt);
        int table_index = compress_builtin_type_id[rdt.get_type_id()];
        if (table_index >= 0) {
            func_ptr = addition_table[table_index];
        } else {
            func_ptr.single = NULL;
            func_ptr.strided = NULL;
        }

        // The signature is (T, T) -> T, so we don't use the original types
        return apply_binary_operator<ckernel_prefix>(ops, rdt, rdt, rdt, func_ptr, "addition");
    } else if (op1dt.get_kind() == string_kind && op2dt.get_kind() == string_kind) {
        ndt::type rdt = ndt::make_string();
        func_ptr.single = &kernels::string_concatenation_kernel::single;
        func_ptr.strided = &kernels::string_concatenation_kernel::strided;
        // The signature is (string, string) -> string, so we don't use the original types
        // NOTE: Using a different name for string concatenation in the generated expression
        return apply_binary_operator<kernels::string_concatenation_kernel>(ops, rdt, rdt, rdt, func_ptr, "string_concat");
    } else {
        stringstream ss;
        ss << "Addition is not supported for dynd types ";
        ss << op1dt << " and " << op2dt;
        throw runtime_error(ss.str());
    }
}

nd::array nd::operator-(const nd::array& op1, const nd::array& op2)
{
    ndt::type rdt;
    expr_operation_pair func_ptr;
    ndt::type op1dt = op1.get_dtype().value_type();
    ndt::type op2dt = op2.get_dtype().value_type();
    if (op1dt.is_builtin() && op1dt.is_builtin()) {
        rdt = promote_types_arithmetic(op1dt, op2dt);
        int table_index = compress_builtin_type_id[rdt.get_type_id()];
        if (table_index >= 0) {
            func_ptr = subtraction_table[table_index];
        }
    }

    nd::array ops[2] = {op1, op2};
    return apply_binary_operator<ckernel_prefix>(ops, rdt, rdt, rdt, func_ptr, "subtraction");
}

nd::array nd::operator*(const nd::array& op1, const nd::array& op2)
{
    ndt::type rdt;
    expr_operation_pair func_ptr;
    ndt::type op1dt = op1.get_dtype().value_type();
    ndt::type op2dt = op2.get_dtype().value_type();
    if (op1dt.is_builtin() && op1dt.is_builtin()) {
        rdt = promote_types_arithmetic(op1dt, op2dt);
        int table_index = compress_builtin_type_id[rdt.get_type_id()];
        if (table_index >= 0) {
            func_ptr = multiplication_table[table_index];
        }
    }

    nd::array ops[2] = {op1, op2};
    return apply_binary_operator<ckernel_prefix>(ops, rdt, rdt, rdt, func_ptr, "multiplication");
}

nd::array nd::operator/(const nd::array& op1, const nd::array& op2)
{
    ndt::type rdt;
    expr_operation_pair func_ptr;
    ndt::type op1dt = op1.get_dtype().value_type();
    ndt::type op2dt = op2.get_dtype().value_type();
    if (op1dt.is_builtin() && op1dt.is_builtin()) {
        rdt = promote_types_arithmetic(op1dt, op2dt);
        int table_index = compress_builtin_type_id[rdt.get_type_id()];
        if (table_index >= 0) {
            func_ptr = division_table[table_index];
        }
    }

    nd::array ops[2] = {op1, op2};
    return apply_binary_operator<ckernel_prefix>(ops, rdt, rdt, rdt, func_ptr, "division");
}
