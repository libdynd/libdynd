//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/buffered_unary_kernels.hpp>
#include "single_assigner_builtin.hpp"

using namespace std;
using namespace dynd;

namespace {
    template<typename dst_type, typename src_type, assign_error_mode errmode>
    struct multiple_assignment_builtin {
        static void strided_assign(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            size_t count, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            for (size_t i = 0; i != count; ++i, dst += dst_stride, src += src_stride) {
                single_assigner_builtin<dst_type, src_type, errmode>::assign(reinterpret_cast<dst_type *>(dst),
                                reinterpret_cast<const src_type *>(src), NULL);
            }
        }
    };
} // anonymous namespace


static unary_operation_pair_t assign_table[builtin_type_id_count][builtin_type_id_count][4] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) unary_operation_pair_t( \
            (unary_single_operation_t)&single_assigner_builtin<dst_type, src_type, errmode>::assign, \
            multiple_assignment_builtin<dst_type, src_type, errmode>::strided_assign)
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_none), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_overflow), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_fractional), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_inexact) \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        ERROR_MODE_LEVEL(dst_type, dynd_bool), \
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

    SRC_TYPE_LEVEL(dynd_bool),
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
#undef SINGLE_OPERATION_PAIR_LEVEL
};

unary_operation_pair_t dynd::get_builtin_dtype_assignment_function(type_id_t dst_type_id, type_id_t src_type_id,
                                                                assign_error_mode errmode)
{
    // Do a table lookup for the built-in range of dtypes
    if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id &&
            src_type_id >= bool_type_id && src_type_id <= complex_float64_type_id &&
            errmode != assign_error_default) {
        return assign_table[dst_type_id][src_type_id][errmode];
    } else {
        return unary_operation_pair_t();
    }
}

void dynd::get_builtin_dtype_assignment_kernel(
                    type_id_t dst_type_id, type_id_t src_type_id,
                    assign_error_mode errmode,
                    const eval::eval_context *ectx,
                    kernel_instance<unary_operation_pair_t>& out_kernel)
{
    // Apply the default error mode from the context if possible
    if (errmode == assign_error_default) {
        if (ectx != NULL) {
            errmode = ectx->default_assign_error_mode;
        } else {
            // Behavior is to return NULL for default mode if no
            // evaluation context is provided
            out_kernel.kernel = unary_operation_pair_t();
            out_kernel.extra.auxdata.free();
            return;
        }
    }

    if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id &&
            src_type_id >= bool_type_id && src_type_id <= complex_float64_type_id &&
            errmode != assign_error_default) {
        out_kernel.kernel = assign_table[dst_type_id][src_type_id][errmode];
        out_kernel.extra.auxdata.free();
    } else {
        stringstream ss;
        ss << "Could not construct dynd assignment kernel from " << dtype(src_type_id) << " to ";
        ss << dtype(dst_type_id) << " with error mode " << errmode;
        throw runtime_error(ss.str());
    }
}

void dynd::get_dtype_assignment_kernel(
                    const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    const eval::eval_context *ectx,
                    kernel_instance<unary_operation_pair_t>& out_kernel)
{
    // special-case matching src and dst dtypes
    if (dst_dt == src_dt) {
        return get_dtype_assignment_kernel(dst_dt, out_kernel);
    }

    // If the casting can be done losslessly, disable the error check to find faster code paths
    if (errmode != assign_error_none && is_lossless_assignment(dst_dt, src_dt)) {
        errmode = assign_error_none;
    }

    if (errmode == assign_error_default && ectx != NULL) {
        errmode = ectx->default_assign_error_mode;
    }

    // Assignment of built-in types
    if (dst_dt.is_builtin() && src_dt.is_builtin()) {
        get_builtin_dtype_assignment_kernel(dst_dt.get_type_id(),
                            src_dt.get_type_id(), errmode, ectx, out_kernel);
        return;
    }

    // Assignment of expression dtypes
    if (src_dt.get_kind() == expression_kind || dst_dt.get_kind() == expression_kind) {
        // Chain the kernels together
        deque<kernel_instance<unary_operation_pair_t>> kernels;
        deque<dtype> dtypes;
        const dtype& src_dt_vdt = src_dt.value_dtype();
        const dtype& dst_dt_vdt = dst_dt.value_dtype();
        //intptr_t next_element_size = 0;

        if (src_dt.get_kind() == expression_kind) {
            // kernel operations from src's storage to value
            push_front_dtype_storage_to_value_kernels(src_dt, ectx, kernels, dtypes);
        }

        if (src_dt_vdt != dst_dt_vdt) {
            // A cast operation from src_dt.value_dtype() to dst_dt
            if (kernels.empty()) {
                dtypes.push_back(src_dt_vdt);
            }
            dtypes.push_back(dst_dt_vdt);
            kernels.push_back(kernel_instance<unary_operation_pair_t>());
            get_dtype_assignment_kernel(dst_dt_vdt, src_dt_vdt,
                                errmode, ectx, kernels.back());
        }

        if (dst_dt.get_kind() == expression_kind) {
            push_back_dtype_value_to_storage_kernels(dst_dt, ectx, kernels, dtypes);
        }

        make_buffered_chain_unary_kernel(kernels, dtypes, out_kernel);
        return;
    }

    if (!dst_dt.is_builtin()) {
        dst_dt.extended()->get_dtype_assignment_kernel(dst_dt, src_dt, errmode, out_kernel);
    } else {
        src_dt.extended()->get_dtype_assignment_kernel(dst_dt, src_dt, errmode, out_kernel);
    }
}

namespace {
    template<class T>
    struct aligned_fixed_size_copy_assign_type {
        static void single(char *dst, const char *src, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            *(T *)dst = *(T *)src;
        }

        static void strided(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride, size_t count, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            for (size_t i = 0; i != count; ++i, dst += dst_stride, src += src_stride) {
                *(T *)dst = *(T *)src;
            }
        }
    };

    template<int N>
    struct aligned_fixed_size_copy_assign;
    template<>
    struct aligned_fixed_size_copy_assign<1> {
        static void single(char *dst, const char *src, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            *dst = *src;
        }

        static void strided(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride, size_t count, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            for (size_t i = 0; i != count; ++i, dst += dst_stride, src += src_stride) {
                *dst = *src;
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
        static void single(char *dst, const char *src, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            memcpy(dst, src, N);
        }

        static void strided(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                        size_t count, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            for (size_t i = 0; i != count; ++i, dst += dst_stride, src += src_stride) {
                memcpy(dst, src, N);
            }
        }
    };
}
static void unaligned_copy_single(char *dst, const char *src, unary_kernel_static_data *extra)
{
    size_t element_size = static_cast<size_t>(get_raw_auxiliary_data(extra->auxdata)>>1);
    memcpy(dst, src, element_size);
}
static void unaligned_copy_strided(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                size_t count, unary_kernel_static_data *extra)
{
    size_t element_size = static_cast<size_t>(get_raw_auxiliary_data(extra->auxdata)>>1);
    for (size_t i = 0; i != count; ++i, dst += dst_stride, src += src_stride) {
        memcpy(dst, src, element_size);
    }
}

void dynd::get_pod_dtype_assignment_kernel(
                    intptr_t element_size, intptr_t alignment,
                    kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (element_size == alignment) {
        // Aligned specialization tables
        switch (element_size) {
            case 1:
                out_kernel.kernel.single = &aligned_fixed_size_copy_assign<1>::single;
                out_kernel.kernel.strided = &aligned_fixed_size_copy_assign<1>::strided;
                break;
            case 2:
                out_kernel.kernel.single = &aligned_fixed_size_copy_assign<2>::single;
                out_kernel.kernel.strided = &aligned_fixed_size_copy_assign<2>::strided;
                break;
            case 4:
                out_kernel.kernel.single = &aligned_fixed_size_copy_assign<4>::single;
                out_kernel.kernel.strided = &aligned_fixed_size_copy_assign<4>::strided;
                break;
            case 8:
                out_kernel.kernel.single = &aligned_fixed_size_copy_assign<8>::single;
                out_kernel.kernel.strided = &aligned_fixed_size_copy_assign<8>::strided;
                break;
            default:
                out_kernel.kernel.single = &unaligned_copy_single;
                out_kernel.kernel.strided = &unaligned_copy_strided;
                break;
        }
    } else {
        // Unaligned specialization tables
        switch (element_size) {
            case 2:
                out_kernel.kernel.single = unaligned_fixed_size_copy_assign<2>::single;
                out_kernel.kernel.strided = unaligned_fixed_size_copy_assign<2>::strided;
                break;
            case 4:
                out_kernel.kernel.single = unaligned_fixed_size_copy_assign<4>::single;
                out_kernel.kernel.strided = unaligned_fixed_size_copy_assign<4>::strided;
                break;
            case 8:
                out_kernel.kernel.single = unaligned_fixed_size_copy_assign<8>::single;
                out_kernel.kernel.strided = unaligned_fixed_size_copy_assign<8>::strided;
                break;
            default:
                out_kernel.kernel.single = &unaligned_copy_single;
                out_kernel.kernel.strided = &unaligned_copy_strided;
                break;
        }
    }
    make_raw_auxiliary_data(out_kernel.extra.auxdata, static_cast<uintptr_t>(element_size)<<1);
}

void dynd::get_dtype_assignment_kernel(const dtype& dt,
                    kernel_instance<unary_operation_pair_t>& out_kernel)
{
    // If the dtype doesn't have a fixed size, get its specific assignment kernel
    if (dt.get_data_size() > 0 && dt.get_memory_management() == pod_memory_management) {
        get_pod_dtype_assignment_kernel(dt.get_data_size(), dt.get_alignment(), out_kernel);
    } else {
        dt.extended()->get_dtype_assignment_kernel(dt, dt, assign_error_none, out_kernel);
    }
}
