//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <sstream>

#include <dnd/ndarray.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/dtype_promotion.hpp>
#include <dnd/arithmetic_op.hpp>
#include "ndarray_expr_node_instances.hpp"

using namespace std;
using namespace dnd;

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

    template<class operation>
    struct loop_general_general_general {
        static void func(char *dst, intptr_t dst_stride,
                                    const char *src0, intptr_t src0_stride,
                                    const char *src1, intptr_t src1_stride,
                                    intptr_t count, const auxiliary_data *)
        {
            typedef typename operation::type T;

            for (intptr_t i = 0; i < count; ++i) {
                *reinterpret_cast<T *>(dst) = operation::operate(*reinterpret_cast<const T *>(src0),
                                                                *reinterpret_cast<const T *>(src1));
                dst += dst_stride;
                src0 += src0_stride;
                src1 += src1_stride;
            }
        }
    };

    template<class operation>
    struct loop_general_stride0_general {
        static void func(char * *dst, intptr_t dst_stride,
                                        const char * *src0, intptr_t,
                                        const char * *src1, intptr_t src1_stride,
                                        intptr_t count, const auxiliary_data *)
        {
            typedef typename operation::type T;

            T src0_value = *reinterpret_cast<const T *>(src0);
            for (intptr_t i = 0; i < count; ++i) {
                *reinterpret_cast<T *>(dst) = operation::operate(src0_value, *reinterpret_cast<const T *>(src1));
                dst += dst_stride;
                src1 += src1_stride;
            }
        }
    };

    template<class operation>
    struct loop_general_general_stride0 {
        static void func(char *dst, intptr_t dst_stride,
                                        const char *src0, intptr_t src0_stride,
                                        const char *src1, intptr_t,
                                        intptr_t count, const auxiliary_data *)
        {
            typedef typename operation::type T;

            T src1_value = *reinterpret_cast<const T *>(src1);
            for (intptr_t i = 0; i < count; ++i) {
                *reinterpret_cast<T *>(dst) = operation::operate(*reinterpret_cast<const T *>(src0), src1_value);
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    };

    template<class operation>
    struct loop_contig_contig_contig {
        static void func(typename operation::type *dst, intptr_t,
                                        const typename operation::type *src0, intptr_t,
                                        const typename operation::type *src1, intptr_t,
                                        intptr_t count, const auxiliary_data *)
        {
            for (intptr_t i = 0; i < count; ++i) {
                //cout << "Inner op c c c " << (void *)dst << " <- " << (void *)src0 << " <oper> " << (void *)src1 << endl;
                //cout << "values " << *src0 << ", " << *src1 << endl;
                *dst = operation::operate(*src0, *src1);
                ++dst;
                ++src0;
                ++src1;
            }
        }
    };

    template<class operation>
    struct loop_contig_stride0_contig {
        static void func(typename operation::type *dst, intptr_t,
                                        const typename operation::type *src0, intptr_t,
                                        const typename operation::type *src1, intptr_t,
                                        intptr_t count, const auxiliary_data *)
        {
            typename operation::type src0_value = *src0;
            for (intptr_t i = 0; i < count; ++i) {
                //cout << "Inner op c s0 c " << (void *)dst << " <- " << (void *)src0 << " <oper> " << (void *)src1 << endl;
                //cout << "values " << *src0 << ", " << *src1 << endl;
                *dst = operation::operate(src0_value, *src1);
                ++dst;
                ++src1;
            }
        }
    };

    template<class operation>
    struct loop_contig_contig_stride0 {
        static void func(typename operation::type *dst, intptr_t,
                                        const typename operation::type *src0, intptr_t,
                                        const typename operation::type *src1, intptr_t,
                                        intptr_t count, const auxiliary_data *)
        {
            typename operation::type src1_value = *src1;
            for (intptr_t i = 0; i < count; ++i) {
                //cout << "Inner op c c s0 " << (void *)dst << " <- " << (void *)src0 << " <oper> " << (void *)src1 << endl;
                //cout << "values " << *src0 << ", " << *src1 << endl;
                *dst = operation::operate(*src0, src1_value);
                ++dst;
                ++src0;
            }
        }
    };

} // anonymous namespace

#define SPECIALIZATION_LEVEL(type, operation) { \
    (binary_operation_t)&loop_general_general_general<operation<type> >::func, \
    (binary_operation_t)&loop_general_stride0_general<operation<type> >::func, \
    (binary_operation_t)&loop_general_general_stride0<operation<type> >::func, \
    (binary_operation_t)&loop_contig_contig_contig<operation<type> >::func, \
    (binary_operation_t)&loop_contig_stride0_contig<operation<type> >::func, \
    (binary_operation_t)&loop_contig_contig_stride0<operation<type> >::func \
    }
#define TYPE_LEVEL(operation) { \
    SPECIALIZATION_LEVEL(int32_t, operation), \
    SPECIALIZATION_LEVEL(int64_t, operation), \
    SPECIALIZATION_LEVEL(uint32_t, operation), \
    SPECIALIZATION_LEVEL(uint64_t, operation), \
    SPECIALIZATION_LEVEL(float, operation), \
    SPECIALIZATION_LEVEL(double, operation) \
    }
#define BUILTIN_OPERATION_TABLE(operation) \
    static binary_operation_t builtin_##operation##_table[6][6] = \
        TYPE_LEVEL(operation)

BUILTIN_OPERATION_TABLE(addition);
BUILTIN_OPERATION_TABLE(subtraction);
BUILTIN_OPERATION_TABLE(multiplication);
BUILTIN_OPERATION_TABLE(division);

#undef BUILTIN_OPERATION_TABLE
#undef TYPE_LEVEL
#undef SPECIALIZATION_LEVEL

typedef binary_operation_t binary_operation_table_t[6];

static binary_operation_t get_builtin_operation_function(
                                binary_operation_table_t *builtin_optable,
                                const dtype& dt, intptr_t dst_stride,
                                intptr_t src0_stride, intptr_t src1_stride)
{
    static int compress_type_id[12] = {-1, -1, -1, -1, 0, 1, -1, -1, 2, 3, 4, 5};
    intptr_t itemsize = dt.itemsize();
    int cid = compress_type_id[dt.type_id()];

    // Pick out a specialized inner loop based on the strides
    if (dst_stride == itemsize) {
        if (src0_stride == itemsize) {
            if (src1_stride == itemsize) {
                return builtin_optable[cid][3];
            } else if (src1_stride == 0) {
                return builtin_optable[cid][5];
            }
        } else if (src0_stride == 0 && src1_stride == itemsize) {
            return builtin_optable[cid][4];
        }
    }

    if (src0_stride == 0) {
        return builtin_optable[cid][1];
    } else if (src1_stride == 0) {
        return builtin_optable[cid][2];
    } else {
        return builtin_optable[cid][0];
    }
}

namespace {

    class binary_operator_factory {
    protected:
        dtype m_dtype;
        binary_operation_table_t *m_builtin_optable;

    public:
        binary_operator_factory() {
        }

        binary_operator_factory(binary_operation_table_t *builtin_optable)
            : m_builtin_optable(builtin_optable)
        {
        }

        void promote_types(const dtype& dt1, const dtype& dt2) {
            m_dtype = promote_dtypes_arithmetic(dt1, dt2);
        }

        const dtype& get_dtype(int) {
            return m_dtype;
        }

        void swap(binary_operator_factory& that) {
            m_dtype.swap(that.m_dtype);
            std::swap(m_builtin_optable, that.m_builtin_optable);
        }

        std::pair<binary_operation_t, std::shared_ptr<auxiliary_data> >
                get_binary_operation(intptr_t dst_fixedstride, intptr_t src1_fixedstride,
                                    intptr_t src2_fixedstride) {
            return std::pair<binary_operation_t, std::shared_ptr<auxiliary_data> >(
                        get_builtin_operation_function(m_builtin_optable,
                                m_dtype, dst_fixedstride,
                                src1_fixedstride, src2_fixedstride),
                        NULL);
        }
    };

} // anonymous namespace

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

    return std::move(result);
}

ndarray dnd::add(const ndarray& op0, const ndarray& op1) {
    return arithmetic_op(op0, op1, builtin_addition_table);
}

ndarray dnd::subtract(const ndarray& op0, const ndarray& op1) {
    return arithmetic_op(op0, op1, builtin_subtraction_table);
}

ndarray dnd::multiply(const ndarray& op0, const ndarray& op1) {
    return arithmetic_op(op0, op1, builtin_multiplication_table);
}

ndarray dnd::divide(const ndarray& op0, const ndarray& op1) {
    return arithmetic_op(op0, op1, builtin_division_table);
}

// These operators are declared in ndarray.hpp

ndarray dnd::operator+(const ndarray& op0, const ndarray& op1) {
    return arithmetic_op(op0, op1, builtin_addition_table);
}

ndarray dnd::operator-(const ndarray& op0, const ndarray& op1) {
    return arithmetic_op(op0, op1, builtin_subtraction_table);
}

ndarray dnd::operator*(const ndarray& op0, const ndarray& op1) {
    return arithmetic_op(op0, op1, builtin_multiplication_table);
}

ndarray dnd::operator/(const ndarray& op0, const ndarray& op1) {
    return arithmetic_op(op0, op1, builtin_division_table);
}
