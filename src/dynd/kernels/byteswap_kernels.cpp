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
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            for (intptr_t i = 0; i < count; ++i) {
                *(T *)dst = byteswap_value(*(T *)src);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *DND_UNUSED(auxdata))
        {
            DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            *(T *)dst = byteswap_value(*(T *)src);
        }

        static void contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            T *dst_cached = reinterpret_cast<T *>(dst);
            const T *src_cached = reinterpret_cast<const T *>(src);

            for (intptr_t i = 0; i < count; ++i) {
                *dst_cached = byteswap_value(*src_cached);

                ++dst_cached;
                ++src_cached;
            }
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            T *dst_cached = reinterpret_cast<T *>(dst);
            const T src_value = byteswap_value(*reinterpret_cast<const T *>(src));

            for (intptr_t i = 0; i < count; ++i) {
                *dst_cached = src_value;

                ++dst_cached;
            }
        }
    };

    template<typename T>
    struct aligned_fixed_size_pairwise_byteswap_kernel {
       static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            for (intptr_t i = 0; i < count; ++i) {
                *(T *)dst = byteswap_value(*(T *)src);
                *((T *)dst + 1) = byteswap_value(*((T *)src + 1));

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *DND_UNUSED(auxdata))
        {
            DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            *(T *)dst = byteswap_value(*(T *)src);
            *((T *)dst + 1) = byteswap_value(*((T *)src + 1));
        }

        static void contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            T *dst_cached = reinterpret_cast<T *>(dst);
            const T *src_cached = reinterpret_cast<const T *>(src);

            for (intptr_t i = 0; i < count; ++i) {
                *dst_cached = byteswap_value(*src_cached);
                ++dst_cached;
                ++src_cached;

                *dst_cached = byteswap_value(*src_cached);
                ++dst_cached;
                ++src_cached;
            }
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *DND_UNUSED(auxdata))
        {
            DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            T *dst_cached = reinterpret_cast<T *>(dst);
            const T src_value = byteswap_value(*reinterpret_cast<const T *>(src));

            for (intptr_t i = 0; i < count; ++i) {
                *dst_cached = src_value;
                ++dst_cached;

                *dst_cached = src_value;
                ++dst_cached;
            }
        }
    };
} // anonymous namespace

/**
 * Byteswaps arbitrary sized data. The auxiliary data should be created by calling
 * make_raw_auxiliary_data(out_auxdata, static_cast<uintptr_t>(element_size)<<1).
 */
static void general_byteswap_kernel(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    intptr_t element_size = static_cast<intptr_t>(get_raw_auxiliary_data(auxdata)>>1);

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
 * Byteswaps arbitrary sized data. The auxiliary data should be created by calling
 * make_raw_auxiliary_data(out_auxdata, static_cast<uintptr_t>(element_size)<<1).
 */
static void general_pairwise_byteswap_kernel(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    intptr_t element_size = static_cast<intptr_t>(get_raw_auxiliary_data(auxdata)>>1);

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

void dynd::get_byteswap_kernel(intptr_t element_size, intptr_t alignment,
                unary_specialization_kernel_instance& out_kernel)
{
    static specialized_unary_operation_table_t aligned_optable[] = {
        {aligned_fixed_size_byteswap<uint16_t>::general_kernel, 
         aligned_fixed_size_byteswap<uint16_t>::scalar_kernel,
         aligned_fixed_size_byteswap<uint16_t>::contiguous_kernel,
         aligned_fixed_size_byteswap<uint16_t>::scalar_to_contiguous_kernel},
        {aligned_fixed_size_byteswap<uint32_t>::general_kernel, 
         aligned_fixed_size_byteswap<uint32_t>::scalar_kernel,
         aligned_fixed_size_byteswap<uint32_t>::contiguous_kernel,
         aligned_fixed_size_byteswap<uint32_t>::scalar_to_contiguous_kernel},
        {aligned_fixed_size_byteswap<uint64_t>::general_kernel, 
         aligned_fixed_size_byteswap<uint64_t>::scalar_kernel,
         aligned_fixed_size_byteswap<uint64_t>::contiguous_kernel,
         aligned_fixed_size_byteswap<uint64_t>::scalar_to_contiguous_kernel}};

    static specialized_unary_operation_table_t general_optable = {
        general_byteswap_kernel,
        general_byteswap_kernel,
        general_byteswap_kernel,
        general_byteswap_kernel};

    if (element_size == alignment) {
        switch (element_size) {
        case 2:
            out_kernel.specializations = aligned_optable[0];
            break;
        case 4:
            out_kernel.specializations = aligned_optable[1];
            break;
        case 8:
            out_kernel.specializations = aligned_optable[2];
            break;
        default:
            out_kernel.specializations = general_optable;
            break;
        }
    } else {
        out_kernel.specializations = general_optable;
    }
    make_raw_auxiliary_data(out_kernel.auxdata, static_cast<uintptr_t>(element_size)<<1);
}

void dynd::get_pairwise_byteswap_kernel(intptr_t element_size, intptr_t alignment,
                unary_specialization_kernel_instance& out_kernel)
{
    if ((element_size&0x01) == 0x01) {
        throw runtime_error("cannot get a pairwise byteswap kernel for an odd-sized dtype");
    }

    static specialized_unary_operation_table_t aligned_optable[] = {
        {aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::general_kernel, 
         aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::scalar_kernel,
         aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::contiguous_kernel,
         aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::scalar_to_contiguous_kernel},
        {aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::general_kernel, 
         aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::scalar_kernel,
         aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::contiguous_kernel,
         aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::scalar_to_contiguous_kernel},
        {aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::general_kernel, 
         aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::scalar_kernel,
         aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::contiguous_kernel,
         aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::scalar_to_contiguous_kernel}};

    static specialized_unary_operation_table_t general_optable = {
        general_pairwise_byteswap_kernel,
        general_pairwise_byteswap_kernel,
        general_pairwise_byteswap_kernel,
        general_pairwise_byteswap_kernel};

    if (element_size <= alignment * 2) {
        switch (element_size) {
        case 4:
            out_kernel.specializations = aligned_optable[0];
            break;
        case 8:
            out_kernel.specializations = aligned_optable[1];
            break;
        case 16:
            out_kernel.specializations = aligned_optable[2];
            break;
        default:
            out_kernel.specializations = general_optable;
            break;
        }
    } else {
        out_kernel.specializations = general_optable;
    }
    make_raw_auxiliary_data(out_kernel.auxdata, static_cast<uintptr_t>(element_size)<<1);
}
