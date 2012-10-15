//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__BINARY_KERNEL_TABLE_HPP_
#define _DND__BINARY_KERNEL_TABLE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/kernels/kernel_instance.hpp>

namespace dynd {

// This is where macros and templates are placed to auto-generate a binary operation
// specialization table. The current version does not work with auxiliary data,
// another version which does should be implemented (probably a policy-based approach
// can provide the best code reuse).
//
// NB: These functions are static member functions because clang insists on optimizing
// them out of existence otherwise

namespace detail {

    template<class OP>
    struct binary_anystride_anystride_anystride_kernel {
        static void func(char *dst, intptr_t dst_stride,
                        const char *src0, intptr_t src0_stride,
                        const char *src1, intptr_t src1_stride,
                        intptr_t count, const AuxDataBase *)
        {
            typedef typename OP::type T;

            for (intptr_t i = 0; i < count; ++i) {
                *reinterpret_cast<T *>(dst) = OP::operate(*reinterpret_cast<const T *>(src0),
                                                                *reinterpret_cast<const T *>(src1));
                dst += dst_stride;
                src0 += src0_stride;
                src1 += src1_stride;
            }
        }
    };

    template<class OP>
    struct binary_anystride_zerostride_anystride_kernel {
        static void func(char * *dst, intptr_t dst_stride,
                        const char * *src0, intptr_t,
                        const char * *src1, intptr_t src1_stride,
                        intptr_t count, const AuxDataBase *)
        {
            typedef typename OP::type T;

            T src0_value = *reinterpret_cast<const T *>(src0);
            for (intptr_t i = 0; i < count; ++i) {
                *reinterpret_cast<T *>(dst) = OP::operate(src0_value, *reinterpret_cast<const T *>(src1));
                dst += dst_stride;
                src1 += src1_stride;
            }
        }
    };

    template<class OP>
    struct binary_anystride_anystride_zerostride_kernel {
    static void func(char *dst, intptr_t dst_stride,
                    const char *src0, intptr_t src0_stride,
                    const char *src1, intptr_t,
                    intptr_t count, const AuxDataBase *)
    {
        typedef typename OP::type T;

        T src1_value = *reinterpret_cast<const T *>(src1);
        for (intptr_t i = 0; i < count; ++i) {
            *reinterpret_cast<T *>(dst) = OP::operate(*reinterpret_cast<const T *>(src0), src1_value);
            dst += dst_stride;
            src0 += src0_stride;
        }
    }
    };

    template<class OP>
    struct binary_contig_contig_contig_kernel {
        static void func(typename OP::type *dst, intptr_t,
                        const typename OP::type *src0, intptr_t,
                        const typename OP::type *src1, intptr_t,
                        intptr_t count, const AuxDataBase *)
        {
            for (intptr_t i = 0; i < count; ++i) {
                //cout << "Inner op c c c " << (void *)dst << " <- " << (void *)src0 << " <oper> " << (void *)src1 << endl;
                //cout << "values " << *src0 << ", " << *src1 << endl;
                *dst = OP::operate(*src0, *src1);
                ++dst;
                ++src0;
                ++src1;
            }
        }
    };

    template<class OP>
    struct binary_contig_zerostride_contig_kernel {
        static void func(typename OP::type *dst, intptr_t,
                        const typename OP::type *src0, intptr_t,
                        const typename OP::type *src1, intptr_t,
                        intptr_t count, const AuxDataBase *)
        {
            typename OP::type src0_value = *src0;
            for (intptr_t i = 0; i < count; ++i) {
                //cout << "Inner op c s0 c " << (void *)dst << " <- " << (void *)src0 << " <oper> " << (void *)src1 << endl;
                //cout << "values " << *src0 << ", " << *src1 << endl;
                *dst = OP::operate(src0_value, *src1);
                ++dst;
                ++src1;
            }
        }
    };

    template<class OP>
    struct binary_contig_contig_zerostride_kernel {
        static void func(typename OP::type *dst, intptr_t,
                        const typename OP::type *src0, intptr_t,
                        const typename OP::type *src1, intptr_t,
                        intptr_t count, const AuxDataBase *)
        {
            typename OP::type src1_value = *src1;
            for (intptr_t i = 0; i < count; ++i) {
                //cout << "Inner op c c s0 " << (void *)dst << " <- " << (void *)src0 << " <oper> " << (void *)src1 << endl;
                //cout << "values " << *src0 << ", " << *src1 << endl;
                *dst = OP::operate(*src0, src1_value);
                ++dst;
                ++src0;
            }
        }
    };

} // namespace detail

#define DND_BUILTIN_DTYPE_BINARY_TABLE_SPECIALIZATION_LEVEL(type, operation) { \
    (binary_operation_t)&detail::binary_anystride_anystride_anystride_kernel<operation<type> >::func, \
    (binary_operation_t)&detail::binary_anystride_zerostride_anystride_kernel<operation<type> >::func, \
    (binary_operation_t)&detail::binary_anystride_anystride_zerostride_kernel<operation<type> >::func, \
    (binary_operation_t)&detail::binary_contig_contig_contig_kernel<operation<type> >::func, \
    (binary_operation_t)&detail::binary_contig_zerostride_contig_kernel<operation<type> >::func, \
    (binary_operation_t)&detail::binary_contig_contig_zerostride_kernel<operation<type> >::func \
    }
#define DND_BUILTIN_DTYPE_BINARY_TABLE_TYPE_LEVEL(operation) { \
    DND_BUILTIN_DTYPE_BINARY_TABLE_SPECIALIZATION_LEVEL(int32_t, operation), \
    DND_BUILTIN_DTYPE_BINARY_TABLE_SPECIALIZATION_LEVEL(int64_t, operation), \
    DND_BUILTIN_DTYPE_BINARY_TABLE_SPECIALIZATION_LEVEL(uint32_t, operation), \
    DND_BUILTIN_DTYPE_BINARY_TABLE_SPECIALIZATION_LEVEL(uint64_t, operation), \
    DND_BUILTIN_DTYPE_BINARY_TABLE_SPECIALIZATION_LEVEL(float, operation), \
    DND_BUILTIN_DTYPE_BINARY_TABLE_SPECIALIZATION_LEVEL(double, operation), \
    DND_BUILTIN_DTYPE_BINARY_TABLE_SPECIALIZATION_LEVEL(complex<float>, operation), \
    DND_BUILTIN_DTYPE_BINARY_TABLE_SPECIALIZATION_LEVEL(complex<double>, operation) \
    }

typedef binary_operation_t specialized_binary_operation_table_t[6];

/**
 * This macro defines a binary operation specialization table for
 * the specified operation, as the variable builtin_<operation>_table.
 * It defines the operation for a default set of built-in types, which
 * excludes the small integers as the intention is for them to promote
 * to bigger integers in typical arithmetic.
 *
 * For example, the following creates builtin_addition table as a static variable:
 *   template<class T>
 *   struct addition {
 *       typedef T type;
 *       static inline T operate(T x, T y) {
 *           return x + y;
 *       }
 *   };
 *   static DND_BUILTIN_DTYPE_BINARY_OPERATION_TABLE(addition);
 */
#define DND_BUILTIN_DTYPE_BINARY_OPERATION_TABLE(operation) \
    dynd::specialized_binary_operation_table_t builtin_##operation##_table[8] = \
        DND_BUILTIN_DTYPE_BINARY_TABLE_TYPE_LEVEL(operation)

/**
 * This returns a specialized binary kernel operation function from
 * a table created by the DND_BUILTIN_DTYPE_BINARY_OPERATION_TABLE macro.
 */
binary_operation_t get_binary_operation_from_builtin_dtype_table(
                                specialized_binary_operation_table_t *builtin_optable,
                                const dtype& dt, intptr_t dst_fixedstride,
                                intptr_t src0_fixedstride, intptr_t src1_fixedstride);


} // namespace dynd

#endif // _DND__BINARY_KERNEL_TABLE_HPP_
