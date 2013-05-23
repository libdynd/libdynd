//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
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
        static void single(char *dst, const char *src,
                        kernel_data_prefix *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            *(T *)dst = byteswap_value(*(T *)src);
        }
        static void strided(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, kernel_data_prefix *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            for (size_t i = 0; i != count; ++i,
                            dst += dst_stride, src += src_stride) {
                *(T *)dst = byteswap_value(*(T *)src);
            }
        }
    };

    template<typename T>
    struct aligned_fixed_size_pairwise_byteswap_kernel {
        static void single(char *dst, const char *src,
                        kernel_data_prefix *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, 0, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            *(T *)dst = byteswap_value(*(T *)src);
            *((T *)dst + 1) = byteswap_value(*((T *)src + 1));
        }
        static void strided(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, kernel_data_prefix *DYND_UNUSED(extra))
        {
            DYND_ASSERT_ALIGNED(dst, dst_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            DYND_ASSERT_ALIGNED(src, src_stride, sizeof(T), "type: " << dynd::dtype(dynd::type_id_of<T>::value));
            for (size_t i = 0; i != count; ++i,
                            dst += dst_stride, src += src_stride) {
                *(T *)dst = byteswap_value(*(T *)src);
                *((T *)dst + 1) = byteswap_value(*((T *)src + 1));
            }
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
                hierarchical_kernel *out, size_t offset_out,
                intptr_t data_size, intptr_t data_alignment,
                kernel_request_t kernreq)
{
    kernel_data_prefix *result = NULL;
    // This is a leaf kernel, so no need to reserve more space
    if (data_size == data_alignment) {
        switch (data_size) {
        case 2:
            result = out->get_at<kernel_data_prefix>(offset_out);
            if (kernreq == kernel_request_single) {
                result->set_function<unary_single_operation_t>(
                                &aligned_fixed_size_byteswap<uint16_t>::single);
            } else if (kernreq == kernel_request_strided) {
                result->set_function<unary_strided_operation_t>(
                                &aligned_fixed_size_byteswap<uint16_t>::strided);
            } else {
                stringstream ss;
                ss << "make_byteswap_assignment_function: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }
            return offset_out + sizeof(kernel_data_prefix);
        case 4:
            result = out->get_at<kernel_data_prefix>(offset_out);
            if (kernreq == kernel_request_single) {
                result->set_function<unary_single_operation_t>(
                                &aligned_fixed_size_byteswap<uint32_t>::single);
            } else if (kernreq == kernel_request_strided) {
                result->set_function<unary_strided_operation_t>(
                                &aligned_fixed_size_byteswap<uint32_t>::strided);
            } else {
                stringstream ss;
                ss << "make_byteswap_assignment_function: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }
            return offset_out + sizeof(kernel_data_prefix);
            break;
        case 8:
            result = out->get_at<kernel_data_prefix>(offset_out);
            if (kernreq == kernel_request_single) {
                result->set_function<unary_single_operation_t>(
                                &aligned_fixed_size_byteswap<uint64_t>::single);
            } else if (kernreq == kernel_request_strided) {
                result->set_function<unary_strided_operation_t>(
                                &aligned_fixed_size_byteswap<uint64_t>::strided);
            } else {
                stringstream ss;
                ss << "make_byteswap_assignment_function: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }
            return offset_out + sizeof(kernel_data_prefix);
            break;
        default:
            break;
        }
    }

    // Use an adapter to a single kernel for this case
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    out->ensure_capacity_leaf(offset_out + sizeof(byteswap_single_kernel_extra));
    result = out->get_at<kernel_data_prefix>(offset_out);
    result->set_function<unary_single_operation_t>(&byteswap_single_kernel_extra::single);
    reinterpret_cast<byteswap_single_kernel_extra *>(result)->data_size = data_size;
    return offset_out + sizeof(byteswap_single_kernel_extra);
}

size_t dynd::make_pairwise_byteswap_assignment_function(
                hierarchical_kernel *out, size_t offset_out,
                intptr_t data_size, intptr_t data_alignment,
                kernel_request_t kernreq)
{
    kernel_data_prefix *result = NULL;
    // This is a leaf kernel, so no need to reserve more space
    if (data_size == data_alignment) {
        switch (data_size) {
        case 4:
            result = out->get_at<kernel_data_prefix>(offset_out);
            if (kernreq == kernel_request_single) {
                result->set_function<unary_single_operation_t>(
                                &aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::single);
            } else if (kernreq == kernel_request_strided) {
                result->set_function<unary_strided_operation_t>(
                                &aligned_fixed_size_pairwise_byteswap_kernel<uint16_t>::strided);
            } else {
                stringstream ss;
                ss << "make_pairwise_byteswap_assignment_function: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }
            return offset_out + sizeof(kernel_data_prefix);
        case 8:
            result = out->get_at<kernel_data_prefix>(offset_out);
            if (kernreq == kernel_request_single) {
                result->set_function<unary_single_operation_t>(
                                &aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::single);
            } else if (kernreq == kernel_request_strided) {
                result->set_function<unary_strided_operation_t>(
                                &aligned_fixed_size_pairwise_byteswap_kernel<uint32_t>::strided);
            } else {
                stringstream ss;
                ss << "make_pairwise_byteswap_assignment_function: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }
            return offset_out + sizeof(kernel_data_prefix);
            break;
        case 16:
            result = out->get_at<kernel_data_prefix>(offset_out);
            if (kernreq == kernel_request_single) {
                result->set_function<unary_single_operation_t>(
                                &aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::single);
            } else if (kernreq == kernel_request_strided) {
                result->set_function<unary_strided_operation_t>(
                                &aligned_fixed_size_pairwise_byteswap_kernel<uint64_t>::strided);
            } else {
                stringstream ss;
                ss << "make_pairwise_byteswap_assignment_function: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }
            return offset_out + sizeof(kernel_data_prefix);
            break;
        default:
            break;
        }
    }

    // Use an adapter to a single kernel for this case
    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
    out->ensure_capacity_leaf(offset_out + sizeof(pairwise_byteswap_single_kernel_extra));
    result = out->get_at<kernel_data_prefix>(offset_out);
    result->set_function<unary_single_operation_t>(&pairwise_byteswap_single_kernel_extra::single);
    reinterpret_cast<pairwise_byteswap_single_kernel_extra *>(result)->data_size = data_size;
    return offset_out + sizeof(pairwise_byteswap_single_kernel_extra);
}
