//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>

#include <dynd/diagnostics.hpp>
#include <dynd/kernels/byteswap_kernels.hpp>

using namespace std;
using namespace dynd;

namespace {

    template<typename T>
    struct aligned_fixed_size_byteswap {
        static void single_kernel(char *dst, const char *src, hierarchical_kernel_common_base *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            *(T *)dst = byteswap_value(*(T *)src);
        }

        static void single(char *dst, const char *src, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            *(T *)dst = byteswap_value(*(T *)src);
        }

        static void strided(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            size_t count, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            for (size_t i = 0; i != count; ++i, dst += dst_stride, src += src_stride) {
                *reinterpret_cast<T *>(dst) = byteswap_value(*reinterpret_cast<const T *>(src));
            }
        }
    };

    template<typename T>
    struct aligned_fixed_size_pairwise_byteswap_kernel {
        static void single_kernel(char *dst, const char *src, hierarchical_kernel_common_base *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            *(T *)dst = byteswap_value(*(T *)src);
            *((T *)dst + 1) = byteswap_value(*((T *)src + 1));
        }

        static void single(char *dst, const char *src, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            *(T *)dst = byteswap_value(*(T *)src);
            *((T *)dst + 1) = byteswap_value(*((T *)src + 1));
        }

        static void strided(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            size_t count, unary_kernel_static_data *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            for (size_t i = 0; i != count; ++i, dst += dst_stride, src += src_stride) {
                reinterpret_cast<T *>(dst)[0] = byteswap_value(reinterpret_cast<const T *>(src)[0]);
                reinterpret_cast<T *>(dst)[1] = byteswap_value(reinterpret_cast<const T *>(src)[1]);
                dst += dst_stride;
                src += src_stride;
            }
        }
    };
} // anonymous namespace

/**
 * Byteswaps arbitrary sized data. The auxiliary data should be created by calling
 * make_raw_auxiliary_data(out_auxdata, static_cast<uintptr_t>(element_size)<<1).
 */
static void general_byteswap_single_kernel(char *dst, const char *src, unary_kernel_static_data *extra)
{
    intptr_t element_size = static_cast<intptr_t>(get_raw_auxiliary_data(extra->auxdata)>>1);

    // Do a different loop for in-place swap versus copying swap,
    // so this one kernel function works correctly for both cases.
    if (src == dst) {
        // In-place swap
        for (intptr_t j = 0; j < element_size/2; ++j) {
            std::swap(dst[j], dst[element_size-j-1]);
        }
    } else {
        for (intptr_t j = 0; j < element_size; ++j) {
            dst[j] = src[element_size-j-1];
        }
    }
}

/**
 * Byteswaps arbitrary sized data. The auxiliary data should be created by calling
 * make_raw_auxiliary_data(out_auxdata, static_cast<uintptr_t>(element_size)<<1).
 */
static void general_pairwise_byteswap_single_kernel(char *dst, const char *src, unary_kernel_static_data *extra)
{
    intptr_t element_size = static_cast<intptr_t>(get_raw_auxiliary_data(extra->auxdata)>>1);

    // Do a different loop for in-place swap versus copying swap,
    // so this one kernel function works correctly for both cases.
    if (src == dst) {
        // In-place swap
        for (intptr_t j = 0; j < element_size/4; ++j) {
            std::swap(dst[j], dst[element_size/2-j-1]);
        }
        for (intptr_t j = 0; j < element_size/4; ++j) {
            std::swap(dst[element_size/2 + j], dst[element_size-j-1]);
        }
    } else {
        for (intptr_t j = 0; j < element_size/2; ++j) {
            dst[j] = src[element_size/2-j-1];
        }
        for (intptr_t j = 0; j < element_size/2; ++j) {
            dst[element_size/2 + j] = src[element_size-j-1];
        }
    }
}

void dynd::get_byteswap_kernel(intptr_t element_size, intptr_t alignment,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (element_size == alignment) {
        switch (element_size) {
        case 2:
            out_kernel.kernel.single = &aligned_fixed_size_byteswap<uint16_t>::single;
            out_kernel.kernel.strided = &aligned_fixed_size_byteswap<uint16_t>::strided;
            break;
        case 4:
            out_kernel.kernel.single = &aligned_fixed_size_byteswap<uint32_t>::single;
            out_kernel.kernel.strided = &aligned_fixed_size_byteswap<uint32_t>::strided;
            break;
        case 8:
            out_kernel.kernel.single = &aligned_fixed_size_byteswap<uint64_t>::single;
            out_kernel.kernel.strided = &aligned_fixed_size_byteswap<uint64_t>::strided;
            break;
        default:
            out_kernel.kernel.single = &general_byteswap_single_kernel;
            out_kernel.kernel.strided = NULL;
            break;
        }
    } else {
        out_kernel.kernel.single = &general_byteswap_single_kernel;
        out_kernel.kernel.strided = NULL;
    }
    make_raw_auxiliary_data(out_kernel.extra.auxdata, static_cast<uintptr_t>(element_size)<<1);
}

void dynd::get_pairwise_byteswap_kernel(intptr_t element_size, intptr_t alignment,
                kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (element_size <= alignment * 2) {
        switch (element_size) {
        case 4:
            out_kernel.kernel.single = &aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::single;
            out_kernel.kernel.strided = &aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::strided;
            break;
        case 8:
            out_kernel.kernel.single = &aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::single;
            out_kernel.kernel.strided = &aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::strided;
            break;
        case 16:
            out_kernel.kernel.single = &aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::single;
            out_kernel.kernel.strided = &aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::strided;
            break;
        default:
            out_kernel.kernel.single = &general_pairwise_byteswap_single_kernel;
            out_kernel.kernel.strided = NULL;
            break;
        }
    } else {
        out_kernel.kernel.single = &general_pairwise_byteswap_single_kernel;
        out_kernel.kernel.strided = NULL;
    }
    make_raw_auxiliary_data(out_kernel.extra.auxdata, static_cast<uintptr_t>(element_size)<<1);
}

// ---------------------- PART BEFORE THIS IS DEPRECATED -------------------------------------

namespace {
    struct byteswap_single_kernel_extra {
        typedef byteswap_single_kernel_extra extra_type;

        hierarchical_kernel_common_base base;
        size_t data_size;

        static void single(char *dst, const char *src, hierarchical_kernel_common_base *extra)
        {
            size_t data_size = reinterpret_cast<extra_type *>(extra)->data_size;
            // Do a different loop for in-place swap versus copying swap,
            // so this one kernel function works correctly for both cases.
            if (src == dst) {
                // In-place swap
                for (intptr_t j = 0; j < data_size/2; ++j) {
                    std::swap(dst[j], dst[data_size-j-1]);
                }
            } else {
                for (intptr_t j = 0; j < data_size; ++j) {
                    dst[j] = src[data_size-j-1];
                }
            }
        }
    };

    struct pairwise_byteswap_single_kernel_extra {
        typedef pairwise_byteswap_single_kernel_extra extra_type;

        hierarchical_kernel_common_base base;
        size_t data_size;

        static void single(char *dst, const char *src, hierarchical_kernel_common_base *extra)
        {
            size_t data_size = reinterpret_cast<extra_type *>(extra)->data_size;
            // Do a different loop for in-place swap versus copying swap,
            // so this one kernel function works correctly for both cases.
            if (src == dst) {
                // In-place swap
                for (intptr_t j = 0; j < data_size/4; ++j) {
                    std::swap(dst[j], dst[data_size/2-j-1]);
                }
                for (intptr_t j = 0; j < data_size/4; ++j) {
                    std::swap(dst[data_size/2 + j], dst[data_size-j-1]);
                }
            } else {
                for (intptr_t j = 0; j < data_size/2; ++j) {
                    dst[j] = src[data_size/2-j-1];
                }
                for (intptr_t j = 0; j < data_size/2; ++j) {
                    dst[data_size/2 + j] = src[data_size-j-1];
                }
            }
        }
    };
} // anonymous namespace

size_t dynd::make_byteswap_assignment_function(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                intptr_t data_size, intptr_t data_alignment)
{
    hierarchical_kernel_common_base *result = NULL;
    // This is a leaf kernel, so no need to reserve more space
    if (data_size == data_alignment) {
        switch (data_size) {
        case 2:
            result = out->get_at<hierarchical_kernel_common_base>(offset_out);
            result->function = &aligned_fixed_size_byteswap<uint16_t>::single_kernel;
            return offset_out + sizeof(hierarchical_kernel_common_base);
        case 4:
            result = out->get_at<hierarchical_kernel_common_base>(offset_out);
            result->function = &aligned_fixed_size_byteswap<uint32_t>::single_kernel;
            return offset_out + sizeof(hierarchical_kernel_common_base);
            break;
        case 8:
            result = out->get_at<hierarchical_kernel_common_base>(offset_out);
            result->function = &aligned_fixed_size_byteswap<uint64_t>::single_kernel;
            return offset_out + sizeof(hierarchical_kernel_common_base);
            break;
        default:
            break;
        }
    }

    // Subtract the base amount to avoid over-reserving memory in this leaf case
    out->ensure_capacity(offset_out + sizeof(byteswap_single_kernel_extra) -
                    sizeof(hierarchical_kernel_common_base));
    result = out->get_at<hierarchical_kernel_common_base>(offset_out);
    result->function = &byteswap_single_kernel_extra::single;
    reinterpret_cast<byteswap_single_kernel_extra *>(result)->data_size = data_size;
    return offset_out + sizeof(byteswap_single_kernel_extra);
}

size_t dynd::make_pairwise_byteswap_assignment_function(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                intptr_t data_size, intptr_t data_alignment)
{
    hierarchical_kernel_common_base *result = NULL;
    // This is a leaf kernel, so no need to reserve more space
    if (data_size == data_alignment) {
        switch (data_size) {
        case 4:
            result = out->get_at<hierarchical_kernel_common_base>(offset_out);
            result->function = &aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::single_kernel;
            return offset_out + sizeof(hierarchical_kernel_common_base);
        case 8:
            result = out->get_at<hierarchical_kernel_common_base>(offset_out);
            result->function = &aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::single_kernel;
            return offset_out + sizeof(hierarchical_kernel_common_base);
            break;
        case 16:
            result = out->get_at<hierarchical_kernel_common_base>(offset_out);
            result->function = &aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::single_kernel;
            return offset_out + sizeof(hierarchical_kernel_common_base);
            break;
        default:
            break;
        }
    }

    // Subtract the base amount to avoid over-reserving memory in this leaf case
    out->ensure_capacity(offset_out + sizeof(pairwise_byteswap_single_kernel_extra) -
                    sizeof(hierarchical_kernel_common_base));
    result = out->get_at<hierarchical_kernel_common_base>(offset_out);
    result->function = &pairwise_byteswap_single_kernel_extra::single;
    reinterpret_cast<pairwise_byteswap_single_kernel_extra *>(result)->data_size = data_size;
    return offset_out + sizeof(pairwise_byteswap_single_kernel_extra);
}
