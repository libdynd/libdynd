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
        static void single(char *dst, const char *src, kernel_data_prefix *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            *(T *)dst = byteswap_value(*(T *)src);
        }
    };

    template<typename T>
    struct aligned_fixed_size_pairwise_byteswap_kernel {
        static void single(char *dst, const char *src, kernel_data_prefix *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            *(T *)dst = byteswap_value(*(T *)src);
            *((T *)dst + 1) = byteswap_value(*((T *)src + 1));
        }
    };
} // anonymous namespace

namespace {
    struct byteswap_single_kernel_extra {
        typedef byteswap_single_kernel_extra extra_type;

        kernel_data_prefix base;
        size_t data_size;

        static void single(char *dst, const char *src, kernel_data_prefix *extra)
        {
            size_t data_size = reinterpret_cast<extra_type *>(extra)->data_size;
            // Do a different loop for in-place swap versus copying swap,
            // so this one kernel function works correctly for both cases.
            if (src == dst) {
                // In-place swap
                for (size_t j = 0; j < data_size/2; ++j) {
                    std::swap(dst[j], dst[data_size-j-1]);
                }
            } else {
                for (size_t j = 0; j < data_size; ++j) {
                    dst[j] = src[data_size-j-1];
                }
            }
        }
    };

    struct pairwise_byteswap_single_kernel_extra {
        typedef pairwise_byteswap_single_kernel_extra extra_type;

        kernel_data_prefix base;
        size_t data_size;

        static void single(char *dst, const char *src, kernel_data_prefix *extra)
        {
            size_t data_size = reinterpret_cast<extra_type *>(extra)->data_size;
            // Do a different loop for in-place swap versus copying swap,
            // so this one kernel function works correctly for both cases.
            if (src == dst) {
                // In-place swap
                for (size_t j = 0; j < data_size/4; ++j) {
                    std::swap(dst[j], dst[data_size/2-j-1]);
                }
                for (size_t j = 0; j < data_size/4; ++j) {
                    std::swap(dst[data_size/2 + j], dst[data_size-j-1]);
                }
            } else {
                for (size_t j = 0; j < data_size/2; ++j) {
                    dst[j] = src[data_size/2-j-1];
                }
                for (size_t j = 0; j < data_size/2; ++j) {
                    dst[data_size/2 + j] = src[data_size-j-1];
                }
            }
        }
    };
} // anonymous namespace

size_t dynd::make_byteswap_assignment_function(
                assignment_kernel *out, size_t offset_out,
                intptr_t data_size, intptr_t data_alignment)
{
    kernel_data_prefix *result = NULL;
    // This is a leaf kernel, so no need to reserve more space
    if (data_size == data_alignment) {
        switch (data_size) {
        case 2:
            result = out->get_at<kernel_data_prefix>(offset_out);
            result->set_function<unary_single_operation_t>(&aligned_fixed_size_byteswap<uint16_t>::single);
            return offset_out + sizeof(kernel_data_prefix);
        case 4:
            result = out->get_at<kernel_data_prefix>(offset_out);
            result->set_function<unary_single_operation_t>(&aligned_fixed_size_byteswap<uint32_t>::single);
            return offset_out + sizeof(kernel_data_prefix);
            break;
        case 8:
            result = out->get_at<kernel_data_prefix>(offset_out);
            result->set_function<unary_single_operation_t>(&aligned_fixed_size_byteswap<uint64_t>::single);
            return offset_out + sizeof(kernel_data_prefix);
            break;
        default:
            break;
        }
    }

    // Subtract the base amount to avoid over-reserving memory in this leaf case
    out->ensure_capacity(offset_out + sizeof(byteswap_single_kernel_extra) -
                    sizeof(kernel_data_prefix));
    result = out->get_at<kernel_data_prefix>(offset_out);
    result->set_function<unary_single_operation_t>(&byteswap_single_kernel_extra::single);
    reinterpret_cast<byteswap_single_kernel_extra *>(result)->data_size = data_size;
    return offset_out + sizeof(byteswap_single_kernel_extra);
}

size_t dynd::make_pairwise_byteswap_assignment_function(
                assignment_kernel *out, size_t offset_out,
                intptr_t data_size, intptr_t data_alignment)
{
    kernel_data_prefix *result = NULL;
    // This is a leaf kernel, so no need to reserve more space
    if (data_size == data_alignment) {
        switch (data_size) {
        case 4:
            result = out->get_at<kernel_data_prefix>(offset_out);
            result->set_function<unary_single_operation_t>(&aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::single);
            return offset_out + sizeof(kernel_data_prefix);
        case 8:
            result = out->get_at<kernel_data_prefix>(offset_out);
            result->set_function<unary_single_operation_t>(&aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::single);
            return offset_out + sizeof(kernel_data_prefix);
            break;
        case 16:
            result = out->get_at<kernel_data_prefix>(offset_out);
            result->set_function<unary_single_operation_t>(&aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::single);
            return offset_out + sizeof(kernel_data_prefix);
            break;
        default:
            break;
        }
    }

    // Subtract the base amount to avoid over-reserving memory in this leaf case
    out->ensure_capacity(offset_out + sizeof(pairwise_byteswap_single_kernel_extra) -
                    sizeof(kernel_data_prefix));
    result = out->get_at<kernel_data_prefix>(offset_out);
    result->set_function<unary_single_operation_t>(&pairwise_byteswap_single_kernel_extra::single);
    reinterpret_cast<pairwise_byteswap_single_kernel_extra *>(result)->data_size = data_size;
    return offset_out + sizeof(pairwise_byteswap_single_kernel_extra);
}
