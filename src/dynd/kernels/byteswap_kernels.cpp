//
// Copyright (C) 2011-15 DyND Developers
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
    static void single(char *dst, char *const *src,
                       ckernel_prefix *DYND_UNUSED(self))
    {
        DYND_ASSERT_ALIGNED(dst, 0, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        DYND_ASSERT_ALIGNED(src, 0, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        *reinterpret_cast<T *>(dst) =
            byteswap_value(**reinterpret_cast<T *const *>(src));
    }
    static void strided(char *dst, intptr_t dst_stride, char *const *src,
                        const intptr_t *src_stride, size_t count,
                        ckernel_prefix *DYND_UNUSED(self))
    {
        DYND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        DYND_ASSERT_ALIGNED(src, src_stride, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
            *reinterpret_cast<T *>(dst) =
                byteswap_value(*reinterpret_cast<T *>(src0));
            dst += dst_stride;
            src0 += src0_stride;
        }
    }
};

template<typename T>
struct aligned_fixed_size_pairwise_byteswap_kernel {
    static void single(char *dst, char *const *src,
                       ckernel_prefix *DYND_UNUSED(self))
    {
        DYND_ASSERT_ALIGNED(dst, 0, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        DYND_ASSERT_ALIGNED(src, 0, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        *reinterpret_cast<T *>(dst) =
            byteswap_value(**reinterpret_cast<T *const *>(src));
        *(reinterpret_cast<T *>(dst) + 1) =
            byteswap_value(*(*reinterpret_cast<T *const *>(src) + 1));
    }
    static void strided(char *dst, intptr_t dst_stride, char *const *src,
                        const intptr_t *src_stride, size_t count,
                        ckernel_prefix *DYND_UNUSED(self))
    {
        DYND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        DYND_ASSERT_ALIGNED(src, src_stride, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
            *reinterpret_cast<T *>(dst) =
                byteswap_value(**reinterpret_cast<T *const *>(src0));
            *(reinterpret_cast<T *>(dst) + 1) =
                byteswap_value(*(*reinterpret_cast<T *const *>(src0) + 1));
            dst += dst_stride;
            src0 += src0_stride;
        }
    }
};
} // anonymous namespace

namespace {
    struct byteswap_ck : public kernels::unary_ck<byteswap_ck> {
        size_t m_data_size;

        inline void single(char *dst, char *src)
        {
            size_t data_size = m_data_size;
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

    struct pairwise_byteswap_ck : public kernels::unary_ck<pairwise_byteswap_ck> {
        size_t m_data_size;

        inline void single(char *dst, char *src)
        {
            size_t data_size = m_data_size;
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
                void *ckb, intptr_t ckb_offset,
                intptr_t data_size, intptr_t data_alignment,
                kernel_request_t kernreq)
{
    ckernel_prefix *result = NULL;
    // This is a leaf kernel, so no need to reserve more space
    if (data_size == data_alignment) {
        switch (data_size) {
        case 2:
            result = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
            result->set_expr_function<aligned_fixed_size_byteswap<uint16_t> >(
                kernreq);
            return ckb_offset;
        case 4:
            result = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
            result->set_expr_function<aligned_fixed_size_byteswap<uint32_t> >(
                kernreq);
            return ckb_offset;
            break;
        case 8:
            result = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
            result->set_expr_function<aligned_fixed_size_byteswap<uint64_t> >(
                kernreq);
            return ckb_offset;
            break;
        default:
            break;
        }
    }

    // Otherwise use the general case ckernel
    byteswap_ck *self = byteswap_ck::create_leaf(ckb, kernreq, ckb_offset);
    self->m_data_size = data_size;
    return ckb_offset;
}

size_t dynd::make_pairwise_byteswap_assignment_function(
                void *ckb, intptr_t ckb_offset,
                intptr_t data_size, intptr_t data_alignment,
                kernel_request_t kernreq)
{
    ckernel_prefix *result = NULL;
    // This is a leaf kernel, so no need to reserve more space
    if (data_size == data_alignment) {
        switch (data_size) {
        case 4:
            result = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
            result->set_expr_function<
                aligned_fixed_size_pairwise_byteswap_kernel<uint16_t> >(
                kernreq);
            return ckb_offset;
        case 8:
            result = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
            result->set_expr_function<
                aligned_fixed_size_pairwise_byteswap_kernel<uint32_t> >(
                kernreq);
            return ckb_offset;
        case 16:
            result = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
            result->set_expr_function<
                aligned_fixed_size_pairwise_byteswap_kernel<uint64_t> >(
                kernreq);
            return ckb_offset;
        default:
            break;
        }
    }

    // Otherwise use the general case ckernel
    pairwise_byteswap_ck *self =
        pairwise_byteswap_ck::create_leaf(ckb, kernreq, ckb_offset);
    self->m_data_size = data_size;
    return ckb_offset;
}
