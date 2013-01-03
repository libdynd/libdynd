//
// Copyright (C) 2011-12, Dynamic NDArray Developers
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
