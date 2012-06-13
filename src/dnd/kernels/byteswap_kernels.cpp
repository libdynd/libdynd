//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>

#include <dnd/diagnostics.hpp>
#include <dnd/kernels/byteswap_kernels.hpp>

using namespace std;
using namespace dnd;

template<class T>
static void aligned_fixed_size_byteswap_kernel(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    intptr_t count, const AuxDataBase *)
{
    DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type id: " << dnd::type_id_of<T>::value);
    DND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type id: " << dnd::type_id_of<T>::value);
    T *dst_cached = reinterpret_cast<T *>(dst);
    const T *src_cached = reinterpret_cast<const T *>(src);
    dst_stride /= sizeof(T);
    src_stride /= sizeof(T);

    for (intptr_t i = 0; i < count; ++i) {
        *dst_cached = byteswap_value(*src_cached);
                
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}

template<class T>
static void aligned_fixed_size_pairwise_byteswap_kernel(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    intptr_t count, const AuxDataBase *)
{
    DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type id: " << type_id_of<T>::value);
    DND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type id: " << type_id_of<T>::value);
    T *dst_cached = reinterpret_cast<T *>(dst);
    const T *src_cached = reinterpret_cast<const T *>(src);
    dst_stride /= sizeof(T);
    src_stride /= sizeof(T);

    for (intptr_t i = 0; i < count; ++i) {
        *dst_cached = byteswap_value(*src_cached);
        *(dst_cached+1) = byteswap_value(*(src_cached+1));
                
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}

/**
 * Byteswaps arbitrary sized data. The auxiliary data should
 * be created by calling make_auxiliary_data<intptr_t>(), and initialized
 * with the element size.
 */
static void general_byteswap_kernel(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    intptr_t element_size = get_auxiliary_data<intptr_t>(auxdata);

    // Do a different loop for in-place swap versus copying swap,
    // so this one kernel function works correctly for both cases.
    if (src == dst && src_stride == dst_stride) {
        // In-place swap
        for (intptr_t i = 0; i < count; ++i) {
            for (intptr_t j = 0; j < element_size/2; ++j) {
                std::swap(dst[j], dst[element_size-j-1]);
            }
                
            dst += dst_stride;
        }
    } else {
        for (intptr_t i = 0; i < count; ++i) {
            for (intptr_t j = 0; j < element_size; ++j) {
                dst[j] = src[element_size-j-1];
            }
                
            src += src_stride;
            dst += dst_stride;
        }
    }
}

/**
 * Byteswaps arbitrary sized data in pairs. The auxiliary data should
 * be created by calling make_auxiliary_data<intptr_t>(), and initialized
 * with the element size.
 */
static void general_pairwise_byteswap_kernel(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    intptr_t element_size = get_auxiliary_data<intptr_t>(auxdata);

    // Do a different loop for in-place swap versus copying swap,
    // so this one kernel function works correctly for both cases.
    if (src == dst && src_stride == dst_stride) {
        // In-place swap
        for (intptr_t i = 0; i < count; ++i) {
            for (intptr_t j = 0; j < element_size/4; ++j) {
                std::swap(dst[j], dst[element_size/2-j-1]);
            }
            for (intptr_t j = 0; j < element_size/4; ++j) {
                std::swap(dst[element_size/2 + j], dst[element_size-j-1]);
            }
                
            dst += dst_stride;
        }
    } else {
        for (intptr_t i = 0; i < count; ++i) {
            for (intptr_t j = 0; j < element_size/2; ++j) {
                dst[j] = src[element_size/2-j-1];
            }
            for (intptr_t j = 0; j < element_size/2; ++j) {
                dst[element_size/2 + j] = src[element_size-j-1];
            }
                
            src += src_stride;
            dst += dst_stride;
        }
    }
}

void dnd::get_byteswap_kernel(intptr_t element_size,
                intptr_t dst_fixedstride, intptr_t src_fixedstride,
                kernel_instance<unary_operation_t>& out_kernel)
{
    switch (element_size) {
    case 2:
        out_kernel.auxdata.free();
        out_kernel.kernel = &aligned_fixed_size_byteswap_kernel<uint16_t>;
        break;
    case 4:
        out_kernel.auxdata.free();
        out_kernel.kernel = &aligned_fixed_size_byteswap_kernel<uint32_t>;
        break;
    case 8:
        out_kernel.auxdata.free();
        out_kernel.kernel = &aligned_fixed_size_byteswap_kernel<uint64_t>;
        break;
    default:
        make_auxiliary_data<intptr_t>(out_kernel.auxdata, element_size);
        out_kernel.kernel = &general_byteswap_kernel;
        break;
    }
}

void dnd::get_pairwise_byteswap_kernel(intptr_t element_size,
                intptr_t dst_fixedstride, intptr_t src_fixedstride,
                kernel_instance<unary_operation_t>& out_kernel)
{
    if ((element_size&0x01) == 0x01) {
        throw runtime_error("cannot get a pairwise byteswap kernel for an odd-sized dtype");
    }

    switch (element_size) {
    case 4:
        out_kernel.auxdata.free();
        out_kernel.kernel = &aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>;
        break;
    case 8:
        out_kernel.auxdata.free();
        out_kernel.kernel = &aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>;
        break;
    case 16:
        out_kernel.auxdata.free();
        out_kernel.kernel = &aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>;
        break;
    default:
        make_auxiliary_data<intptr_t>(out_kernel.auxdata, element_size);
        out_kernel.kernel = &general_pairwise_byteswap_kernel;
        break;
    }
}