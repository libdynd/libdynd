//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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
    static void single(char *dst, const char *const *src,
                       ckernel_prefix *DYND_UNUSED(self))
    {
        DYND_ASSERT_ALIGNED(dst, 0, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        DYND_ASSERT_ALIGNED(src, 0, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        *reinterpret_cast<T *>(dst) =
            byteswap_value(**reinterpret_cast<const T *const *>(src));
    }
    static void strided(char *dst, intptr_t dst_stride, const char *const *src,
                        const intptr_t *src_stride, size_t count,
                        ckernel_prefix *DYND_UNUSED(self))
    {
        DYND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        DYND_ASSERT_ALIGNED(src, src_stride, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
            *reinterpret_cast<T *>(dst) =
                byteswap_value(*reinterpret_cast<const T *>(src0));
            dst += dst_stride;
            src0 += src0_stride;
        }
    }
};

template<typename T>
struct aligned_fixed_size_pairwise_byteswap_kernel {
    static void single(char *dst, const char *const *src,
                       ckernel_prefix *DYND_UNUSED(self))
    {
        DYND_ASSERT_ALIGNED(dst, 0, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        DYND_ASSERT_ALIGNED(src, 0, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        *reinterpret_cast<T *>(dst) =
            byteswap_value(**reinterpret_cast<const T *const *>(src));
        *(reinterpret_cast<T *>(dst) + 1) =
            byteswap_value(*(*reinterpret_cast<const T *const *>(src) + 1));
    }
    static void strided(char *dst, intptr_t dst_stride, const char *const *src,
                        const intptr_t *src_stride, size_t count,
                        ckernel_prefix *DYND_UNUSED(self))
    {
        DYND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        DYND_ASSERT_ALIGNED(src, src_stride, sizeof(T),
                            "type: " << ndt::type(dynd::type_id_of<T>::value));
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
            *reinterpret_cast<T *>(dst) =
                byteswap_value(**reinterpret_cast<const T *const *>(src0));
            *(reinterpret_cast<T *>(dst) + 1) =
                byteswap_value(*(*reinterpret_cast<const T *const *>(src0) + 1));
            dst += dst_stride;
            src0 += src0_stride;
        }
    }
};
} // anonymous namespace

namespace {
    struct byteswap_ck : public kernels::unary_ck<byteswap_ck> {
        size_t m_data_size;

        inline void single(char *dst, const char *src)
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

        inline void single(char *dst, const char *src)
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
                ckernel_builder *out_ckb, size_t ckb_offset,
                intptr_t data_size, intptr_t data_alignment,
                kernel_request_t kernreq)
{
    ckernel_prefix *result = NULL;
    // This is a leaf kernel, so no need to reserve more space
    if (data_size == data_alignment) {
        switch (data_size) {
        case 2:
            result = out_ckb->get_at<ckernel_prefix>(ckb_offset);
            result->set_expr_function<aligned_fixed_size_byteswap<uint16_t> >(
                kernreq);
            return ckb_offset + sizeof(ckernel_prefix);
        case 4:
            result = out_ckb->get_at<ckernel_prefix>(ckb_offset);
            result->set_expr_function<aligned_fixed_size_byteswap<uint32_t> >(
                kernreq);
            return ckb_offset + sizeof(ckernel_prefix);
            break;
        case 8:
            result = out_ckb->get_at<ckernel_prefix>(ckb_offset);
            result->set_expr_function<aligned_fixed_size_byteswap<uint64_t> >(
                kernreq);
            return ckb_offset + sizeof(ckernel_prefix);
            break;
        default:
            break;
        }
    }

    // Otherwise use the general case ckernel
    byteswap_ck *self = byteswap_ck::create_leaf(out_ckb, ckb_offset, kernreq);
    self->m_data_size = data_size;
    return ckb_offset + sizeof(byteswap_ck);
}

size_t dynd::make_pairwise_byteswap_assignment_function(
                ckernel_builder *out_ckb, size_t ckb_offset,
                intptr_t data_size, intptr_t data_alignment,
                kernel_request_t kernreq)
{
    ckernel_prefix *result = NULL;
    // This is a leaf kernel, so no need to reserve more space
    if (data_size == data_alignment) {
        switch (data_size) {
        case 4:
            result = out_ckb->get_at<ckernel_prefix>(ckb_offset);
            result->set_expr_function<
                aligned_fixed_size_pairwise_byteswap_kernel<uint16_t> >(
                kernreq);
            return ckb_offset + sizeof(ckernel_prefix);
        case 8:
            result = out_ckb->get_at<ckernel_prefix>(ckb_offset);
            result->set_expr_function<
                aligned_fixed_size_pairwise_byteswap_kernel<uint32_t> >(
                kernreq);
            return ckb_offset + sizeof(ckernel_prefix);
        case 16:
            result = out_ckb->get_at<ckernel_prefix>(ckb_offset);
            result->set_expr_function<
                aligned_fixed_size_pairwise_byteswap_kernel<uint64_t> >(
                kernreq);
            return ckb_offset + sizeof(ckernel_prefix);
        default:
            break;
        }
    }

    // Otherwise use the general case ckernel
    pairwise_byteswap_ck *self =
        pairwise_byteswap_ck::create_leaf(out_ckb, ckb_offset, kernreq);
    self->m_data_size = data_size;
    return ckb_offset + sizeof(byteswap_ck);
}
