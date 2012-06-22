//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtype.hpp>
#include <dnd/kernels/assignment_kernels.hpp>
#include <dnd/kernels/buffered_unary_kernels.hpp>
#include "single_assigner_builtin.hpp"

using namespace std;
using namespace dnd;

static assignment_function_t single_assign_table[builtin_type_id_count][builtin_type_id_count][4] =
{
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        (assignment_function_t)&single_assigner_builtin<dst_type, src_type, assign_error_none>::assign, \
        (assignment_function_t)&single_assigner_builtin<dst_type, src_type, assign_error_overflow>::assign, \
        (assignment_function_t)&single_assigner_builtin<dst_type, src_type, assign_error_fractional>::assign, \
        (assignment_function_t)&single_assigner_builtin<dst_type, src_type, assign_error_inexact>::assign \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        ERROR_MODE_LEVEL(dst_type, dnd_bool), \
        ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t), \
        ERROR_MODE_LEVEL(dst_type, int32_t), \
        ERROR_MODE_LEVEL(dst_type, int64_t), \
        ERROR_MODE_LEVEL(dst_type, uint8_t), \
        ERROR_MODE_LEVEL(dst_type, uint16_t), \
        ERROR_MODE_LEVEL(dst_type, uint32_t), \
        ERROR_MODE_LEVEL(dst_type, uint64_t), \
        ERROR_MODE_LEVEL(dst_type, float), \
        ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, complex<float>), \
        ERROR_MODE_LEVEL(dst_type, complex<double>) \
    }

    SRC_TYPE_LEVEL(dnd_bool),
    SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),
    SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),
    SRC_TYPE_LEVEL(uint8_t),
    SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),
    SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(float),
    SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(complex<float>),
    SRC_TYPE_LEVEL(complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
};

assignment_function_t dnd::get_builtin_dtype_assignment_function(type_id_t dst_type_id, type_id_t src_type_id,
                                                                assign_error_mode errmode)
{
    // Do a table lookup for the built-in range of dtypes
    if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id &&
            src_type_id >= bool_type_id && src_type_id <= complex_float64_type_id) {
        return single_assign_table[dst_type_id][src_type_id][errmode];
    } else {
        return NULL;
    }
}

void dnd::multiple_assignment_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                                    intptr_t count, const AuxDataBase *auxdata)
{
    assignment_function_t asn = reinterpret_cast<assignment_function_t>(get_raw_auxiliary_data(auxdata)&~1);


    char *dst_cached = reinterpret_cast<char *>(dst);
    const char *src_cached = reinterpret_cast<const char *>(src);

    for (intptr_t i = 0; i < count; ++i) {
        asn(dst_cached, src_cached);
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}

namespace {
    template<typename dst_type, typename src_type, assign_error_mode errmode>
    struct multiple_assignment_builtin {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            for (intptr_t i = 0; i < count; ++i) {
                single_assigner_builtin<dst_type, src_type, errmode>::assign(
                                reinterpret_cast<dst_type *>(dst), reinterpret_cast<const src_type *>(src));

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *DND_UNUSED(auxdata))
        {
            single_assigner_builtin<dst_type, src_type, errmode>::assign(
                            reinterpret_cast<dst_type *>(dst), reinterpret_cast<const src_type *>(src));
        }

        static void contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);
            const src_type *src_cached = reinterpret_cast<const src_type *>(src);

            for (intptr_t i = 0; i < count; ++i) {
                single_assigner_builtin<dst_type, src_type, errmode>::assign(dst_cached, src_cached);

                ++dst_cached;
                ++src_cached;
            }
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);
            const src_type src_value = *reinterpret_cast<const src_type *>(src);

            for (intptr_t i = 0; i < count; ++i) {
                single_assigner_builtin<dst_type, src_type, errmode>::assign(dst_cached, &src_value);

                ++dst_cached;
            }
        }
    };
} // anonymous namespace

#define DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, src_type) \
    { \
        multiple_assignment_builtin<dst_type, src_type, errmode>::general_kernel, \
        multiple_assignment_builtin<dst_type, src_type, errmode>::scalar_kernel, \
        multiple_assignment_builtin<dst_type, src_type, errmode>::contiguous_kernel, \
        multiple_assignment_builtin<dst_type, src_type, errmode>::scalar_to_contiguous_kernel \
    }

#define DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, dst_type) \
    { \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, dnd_bool), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, int8_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, int16_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, int32_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, int64_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, uint8_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, uint16_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, uint32_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, uint64_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, float), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, double), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, complex<float>), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SPECIALIZATION_LEVEL(errmode, dst_type, complex<double>) \
    }

#define DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE(errmode) \
    { \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, dnd_bool), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, int8_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, int16_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, int32_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, int64_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, uint8_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, uint16_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, uint32_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, uint64_t), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, float), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, double), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, complex<float>), \
        DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE_SRC_TYPE_LEVEL(errmode, complex<double>) \
    }


void dnd::get_builtin_dtype_assignment_kernel(
                    type_id_t dst_type_id, type_id_t src_type_id,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel)
{
    if (errmode == assign_error_fractional) {
        // The default error mode is fractional, so do specializations for it.
        static specialized_unary_operation_table_t fractional_optable[builtin_type_id_count][builtin_type_id_count] =
                DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE(assign_error_fractional);

        out_kernel.specializations = fractional_optable[dst_type_id][src_type_id];
        // Make sure there's no stray auxiliary data
        out_kernel.auxdata.free();
    } else if (errmode == assign_error_none) {
        // The no-checking error mode also gets specializations, as it's intended for speed
        static specialized_unary_operation_table_t fractional_optable[builtin_type_id_count][builtin_type_id_count] =
                DTYPE_ASSIGN_BUILTIN_KERNEL_TABLE(assign_error_none);

        out_kernel.specializations = fractional_optable[dst_type_id][src_type_id];
        // Make sure there's no stray auxiliary data
        out_kernel.auxdata.free();
    } else {
        // Use a multiple assignment kernel with a assignment function for all the other cases.
        static specialized_unary_operation_table_t fn_optable = {
            &multiple_assignment_kernel,
            &multiple_assignment_kernel,
            &multiple_assignment_kernel,
            &multiple_assignment_kernel};
        assignment_function_t asn = get_builtin_dtype_assignment_function(dst_type_id, src_type_id, errmode);
        if (asn != NULL) {
            out_kernel.specializations = fn_optable;
            make_raw_auxiliary_data(out_kernel.auxdata, reinterpret_cast<uintptr_t>(asn));
            return;
        }
    }
}

void dnd::get_dtype_assignment_kernel(
                    const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel)
{
    // special-case matching src and dst dtypes
    if (dst_dt == src_dt) {
        return get_dtype_assignment_kernel(dst_dt, out_kernel);
    }

    // If the casting can be done losslessly, disable the error check to find faster code paths
    if (errmode != assign_error_none && is_lossless_assignment(dst_dt, src_dt)) {
        errmode = assign_error_none;
    }

    // Assignment of built-in types
    if (dst_dt.extended() == NULL && src_dt.extended() == NULL) {
        get_builtin_dtype_assignment_kernel(dst_dt.type_id(),
                            src_dt.type_id(), errmode, out_kernel);
        return;
    }

    // Assignment of expression dtypes
    if (src_dt.kind() == expression_kind || dst_dt.kind() == expression_kind) {
        // Chain the kernels together
        deque<unary_specialization_kernel_instance> kernels;
        deque<intptr_t> element_sizes;
        const dtype& src_dt_vdt = src_dt.value_dtype();
        const dtype& dst_dt_vdt = dst_dt.value_dtype();
        intptr_t next_element_size = 0;

        if (src_dt.kind() == expression_kind) {
            // kernel operations from src's storage to value
            push_front_dtype_storage_to_value_kernels(src_dt, kernels, element_sizes);
            next_element_size = src_dt_vdt.element_size();
        }

        if (src_dt_vdt != dst_dt_vdt) {
            // A cast operation from src_dt.value_dtype() to dst_dt
            kernels.push_back(unary_specialization_kernel_instance());
            get_dtype_assignment_kernel(dst_dt_vdt, src_dt_vdt,
                                errmode, kernels.back());
            if (next_element_size != 0) {
                element_sizes.push_back(next_element_size);
            }
            next_element_size = dst_dt_vdt.element_size();
        }

        if (dst_dt.kind() == expression_kind) {
            if (next_element_size != 0) {
                element_sizes.push_back(next_element_size);
            }
            push_back_dtype_value_to_storage_kernels(dst_dt, kernels, element_sizes);
        }

        make_buffered_chain_unary_kernel(kernels, element_sizes, out_kernel);
        return;
    }

    stringstream ss;
    ss << "strided assignment from " << src_dt << " to " << dst_dt << " isn't yet supported";
    throw std::runtime_error(ss.str());
}

namespace {
    template<class T>
    struct aligned_fixed_size_copy_assign_type {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            for (intptr_t i = 0; i < count; ++i) {
                *(T *)dst = *(T *)src;

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *DND_UNUSED(auxdata))
        {
            *(T *)dst = *(T *)src;
        }

        static void contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            T *dst_cached = reinterpret_cast<T *>(dst);
            const T *src_cached = reinterpret_cast<const T *>(src);

            for (intptr_t i = 0; i < count; ++i) {
                *dst_cached = *src_cached;

                ++dst_cached;
                ++src_cached;
            }
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            T *dst_cached = reinterpret_cast<T *>(dst);
            const T src_value = *reinterpret_cast<const T *>(src);

            for (intptr_t i = 0; i < count; ++i) {
                *dst_cached = src_value;

                ++dst_cached;
            }
        }
    };

    template<int N>
    struct aligned_fixed_size_copy_assign;
    template<>
    struct aligned_fixed_size_copy_assign<1> {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            for (intptr_t i = 0; i < count; ++i) {
                *dst = *src;

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *DND_UNUSED(auxdata))
        {
            *dst = *src;
        }

        static void contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            memcpy(dst, src, count);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            char src_value = *src;
            for (intptr_t i = 0; i < count; ++i) {
                *dst = src_value;

                ++dst;
            }
        }
    };
    template<>
    struct aligned_fixed_size_copy_assign<2> : public aligned_fixed_size_copy_assign_type<int16_t> {};
    template<>
    struct aligned_fixed_size_copy_assign<4> : public aligned_fixed_size_copy_assign_type<int32_t> {};
    template<>
    struct aligned_fixed_size_copy_assign<8> : public aligned_fixed_size_copy_assign_type<int64_t> {};

    template<int N>
    struct unaligned_fixed_size_copy_assign {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, src, N);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *DND_UNUSED(auxdata))
        {
            memcpy(dst, src, N);
        }

        static void contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            memcpy(dst, src, count * N);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            for (intptr_t i = 0; i < count; ++i) {
                memcpy(dst, src, N);

                dst += N;
            }
        }
    };
}
static void unaligned_scalar_copy_assign_kernel(char *dst, intptr_t, const char *src, intptr_t,
                            intptr_t DND_UNUSED(count), const AuxDataBase *auxdata)
{
    intptr_t element_size = static_cast<intptr_t>(get_raw_auxiliary_data(auxdata)>>1);
    memcpy(dst, src, element_size);
}
static void unaligned_contig_copy_assign_kernel(char *dst, intptr_t, const char *src, intptr_t,
                            intptr_t count, const AuxDataBase *auxdata)
{
    intptr_t element_size = static_cast<intptr_t>(get_raw_auxiliary_data(auxdata)>>1);
    memcpy(dst, src, element_size * count);
}

static void unaligned_strided_copy_assign_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
{
    char *dst_cached = reinterpret_cast<char *>(dst);
    const char *src_cached = reinterpret_cast<const char *>(src);
    intptr_t element_size = static_cast<intptr_t>(get_raw_auxiliary_data(auxdata)>>1);

    for (intptr_t i = 0; i < count; ++i) {
        memcpy(dst_cached, src_cached, element_size);
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}

void dnd::get_pod_dtype_assignment_kernel(
                    intptr_t element_size, intptr_t alignment,
                    unary_specialization_kernel_instance& out_kernel)
{
    // Aligned size-based specialization tables
    static specialized_unary_operation_table_t aligned_optable[] = {
        {aligned_fixed_size_copy_assign<1>::general_kernel,
         aligned_fixed_size_copy_assign<1>::scalar_kernel,
         aligned_fixed_size_copy_assign<1>::contiguous_kernel,
         aligned_fixed_size_copy_assign<1>::scalar_to_contiguous_kernel},
        {aligned_fixed_size_copy_assign<2>::general_kernel,
         aligned_fixed_size_copy_assign<2>::scalar_kernel,
         aligned_fixed_size_copy_assign<2>::contiguous_kernel,
         aligned_fixed_size_copy_assign<2>::scalar_to_contiguous_kernel},
        {aligned_fixed_size_copy_assign<4>::general_kernel,
         aligned_fixed_size_copy_assign<4>::scalar_kernel,
         aligned_fixed_size_copy_assign<4>::contiguous_kernel,
         aligned_fixed_size_copy_assign<4>::scalar_to_contiguous_kernel},
        {aligned_fixed_size_copy_assign<8>::general_kernel,
         aligned_fixed_size_copy_assign<8>::scalar_kernel,
         aligned_fixed_size_copy_assign<8>::contiguous_kernel,
         aligned_fixed_size_copy_assign<8>::scalar_to_contiguous_kernel}};
    static specialized_unary_operation_table_t unaligned_optable[] = {
        {unaligned_fixed_size_copy_assign<2>::general_kernel,
         unaligned_fixed_size_copy_assign<2>::scalar_kernel,
         unaligned_fixed_size_copy_assign<2>::contiguous_kernel,
         unaligned_fixed_size_copy_assign<2>::scalar_to_contiguous_kernel},
        {unaligned_fixed_size_copy_assign<4>::general_kernel,
         unaligned_fixed_size_copy_assign<4>::scalar_kernel,
         unaligned_fixed_size_copy_assign<4>::contiguous_kernel,
         unaligned_fixed_size_copy_assign<4>::scalar_to_contiguous_kernel},
        {unaligned_fixed_size_copy_assign<8>::general_kernel,
         unaligned_fixed_size_copy_assign<8>::scalar_kernel,
         unaligned_fixed_size_copy_assign<8>::contiguous_kernel,
         unaligned_fixed_size_copy_assign<8>::scalar_to_contiguous_kernel}};
    // Generic specialization table
    static specialized_unary_operation_table_t general_optable = {
        unaligned_strided_copy_assign_kernel,
        unaligned_scalar_copy_assign_kernel,
        unaligned_contig_copy_assign_kernel,
        unaligned_strided_copy_assign_kernel};

    if (element_size == alignment) {
        // Aligned specialization tables
        switch (element_size) {
            case 1:
                out_kernel.specializations = aligned_optable[0];
                break;
            case 2:
                out_kernel.specializations = aligned_optable[1];
                break;
            case 4:
                out_kernel.specializations = aligned_optable[2];
                break;
            case 8:
                out_kernel.specializations = aligned_optable[3];
                break;
            default:
                out_kernel.specializations = general_optable;
                break;
        }
    } else {
        // Unaligned specialization tables
        switch (element_size) {
            case 2:
                out_kernel.specializations = unaligned_optable[0];
                break;
            case 4:
                out_kernel.specializations = unaligned_optable[1];
                break;
            case 8:
                out_kernel.specializations = unaligned_optable[2];
                break;
            default:
                out_kernel.specializations = general_optable;
                break;
        }
    }
    make_raw_auxiliary_data(out_kernel.auxdata, static_cast<uintptr_t>(element_size)<<1);
}

void dnd::get_dtype_assignment_kernel(const dtype& dt,
                    unary_specialization_kernel_instance& out_kernel)
{
    if (!dt.is_object_type()) {
        get_pod_dtype_assignment_kernel(dt.element_size(), dt.alignment(),
                            out_kernel);
        return;
    } else {
        if (dt.kind() == expression_kind) {
            // In the case of an expression dtype, just copy the storage
            // directly instead of chaining multiple casting operations
            // together.
            get_dtype_assignment_kernel(dt.storage_dtype(),
                            out_kernel);
            return;
        }

        throw std::runtime_error("cannot assign object dtypes yet");
    }
}
