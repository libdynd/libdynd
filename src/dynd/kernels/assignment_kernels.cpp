//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include "single_assigner_builtin.hpp"

using namespace std;
using namespace dynd;

namespace {
    template<class T>
    struct aligned_fixed_size_copy_assign_type {
        static void single(char *dst, const char *src,
                        ckernel_prefix *DYND_UNUSED(extra))
        {
            *(T *)dst = *(T *)src;
        }

        static void strided(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(extra))
        {
            for (size_t i = 0; i != count; ++i,
                            dst += dst_stride, src += src_stride) {
                *(T *)dst = *(T *)src;
            }
        }
    };

    template<int N>
    struct aligned_fixed_size_copy_assign;
    template<>
    struct aligned_fixed_size_copy_assign<1> {
        static void single(char *dst, const char *src,
                            ckernel_prefix *DYND_UNUSED(extra))
        {
            *dst = *src;
        }

        static void strided(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(extra))
        {
            for (size_t i = 0; i != count; ++i,
                            dst += dst_stride, src += src_stride) {
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
        static void single(char *dst, const char *src,
                        ckernel_prefix *DYND_UNUSED(extra))
        {
            memcpy(dst, src, N);
        }

        static void strided(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(extra))
        {
            for (size_t i = 0; i != count; ++i,
                            dst += dst_stride, src += src_stride) {
                memcpy(dst, src, N);
            }
        }
    };
}
struct unaligned_copy_single_kernel_extra {
    ckernel_prefix base;
    size_t data_size;
};
static void unaligned_copy_single(char *dst, const char *src,
                ckernel_prefix *extra)
{
    size_t data_size = reinterpret_cast<unaligned_copy_single_kernel_extra *>(extra)->data_size;
    memcpy(dst, src, data_size);
}
static void unaligned_copy_strided(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, ckernel_prefix *extra)
{
    size_t data_size = reinterpret_cast<unaligned_copy_single_kernel_extra *>(extra)->data_size;
    for (size_t i = 0; i != count; ++i,
                    dst += dst_stride, src += src_stride) {
        memcpy(dst, src, data_size);
    }
}

size_t dynd::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx)
{
    if (errmode == assign_error_default && ectx != NULL) {
        errmode = ectx->default_assign_error_mode;
    }

    if (dst_tp.is_builtin()) {
        if (src_tp.is_builtin()) {
            // If the casting can be done losslessly, disable the error check to find faster code paths
            if (errmode != assign_error_none && is_lossless_assignment(dst_tp, src_tp)) {
                errmode = assign_error_none;
            }

            if (dst_tp.extended() == src_tp.extended()) {
                return make_pod_typed_data_assignment_kernel(out, offset_out,
                                dst_tp.get_data_size(), dst_tp.get_data_alignment(),
                                kernreq);
            } else {
                return make_builtin_type_assignment_kernel(out, offset_out,
                                dst_tp.get_type_id(), src_tp.get_type_id(),
                                kernreq, errmode);
            }
        } else {
            return src_tp.extended()->make_assignment_kernel(out, offset_out,
                            dst_tp, dst_metadata,
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        }
    } else {
        return dst_tp.extended()->make_assignment_kernel(out, offset_out,
                        dst_tp, dst_metadata,
                        src_tp, src_metadata,
                        kernreq, errmode, ectx);
    }
}

size_t dynd::make_pod_typed_data_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                size_t data_size, size_t data_alignment,
                kernel_request_t kernreq)
{
    bool single = (kernreq == kernel_request_single);
    if (!single && kernreq != kernel_request_strided) {
        stringstream ss;
        ss << "make_pod_typed_data_assignment_kernel: unrecognized request " << (int)kernreq;
        throw runtime_error(ss.str());
    }
    ckernel_prefix *result = NULL;
    if (data_size == data_alignment) {
        // Aligned specialization tables
        // No need to reserve more space in the trivial cases, the space for a leaf is already there
        switch (data_size) {
            case 1:
                result = out->get_at<ckernel_prefix>(offset_out);
                if (single) {
                    result->set_function<unary_single_operation_t>(
                                    &aligned_fixed_size_copy_assign<1>::single);
                } else {
                    result->set_function<unary_strided_operation_t>(
                                    &aligned_fixed_size_copy_assign<1>::strided);
                }
                return offset_out + sizeof(ckernel_prefix);
            case 2:
                result = out->get_at<ckernel_prefix>(offset_out);
                if (single) {
                    result->set_function<unary_single_operation_t>(
                                    &aligned_fixed_size_copy_assign<2>::single);
                } else {
                    result->set_function<unary_strided_operation_t>(
                                    &aligned_fixed_size_copy_assign<2>::strided);
                }
                return offset_out + sizeof(ckernel_prefix);
            case 4:
                result = out->get_at<ckernel_prefix>(offset_out);
                if (single) {
                    result->set_function<unary_single_operation_t>(
                                    &aligned_fixed_size_copy_assign<4>::single);
                } else {
                    result->set_function<unary_strided_operation_t>(
                                    &aligned_fixed_size_copy_assign<4>::strided);
                }
                return offset_out + sizeof(ckernel_prefix);
            case 8:
                result = out->get_at<ckernel_prefix>(offset_out);
                if (single) {
                    result->set_function<unary_single_operation_t>(
                                    &aligned_fixed_size_copy_assign<8>::single);
                } else {
                    result->set_function<unary_strided_operation_t>(
                                    &aligned_fixed_size_copy_assign<8>::strided);
                }
                return offset_out + sizeof(ckernel_prefix);
            default:
                out->ensure_capacity_leaf(offset_out + sizeof(unaligned_copy_single_kernel_extra));
                result = out->get_at<ckernel_prefix>(offset_out);
                if (single) {
                    result->set_function<unary_single_operation_t>(&unaligned_copy_single);
                } else {
                    result->set_function<unary_strided_operation_t>(&unaligned_copy_strided);
                }
                reinterpret_cast<unaligned_copy_single_kernel_extra *>(result)->data_size = data_size;
                return offset_out + sizeof(unaligned_copy_single_kernel_extra);
        }
    } else {
        // Unaligned specialization tables
        switch (data_size) {
            case 2:
                result = out->get_at<ckernel_prefix>(offset_out);
                if (single) {
                    result->set_function<unary_single_operation_t>(
                                    unaligned_fixed_size_copy_assign<2>::single);
                } else {
                    result->set_function<unary_strided_operation_t>(
                                    unaligned_fixed_size_copy_assign<2>::strided);
                }
                return offset_out + sizeof(ckernel_prefix);
            case 4:
                result = out->get_at<ckernel_prefix>(offset_out);
                if (single) {
                    result->set_function<unary_single_operation_t>(
                                    unaligned_fixed_size_copy_assign<4>::single);
                } else {
                    result->set_function<unary_strided_operation_t>(
                                    unaligned_fixed_size_copy_assign<4>::strided);
                }
                return offset_out + sizeof(ckernel_prefix);
            case 8:
                result = out->get_at<ckernel_prefix>(offset_out);
                if (single) {
                    result->set_function<unary_single_operation_t>(
                                    unaligned_fixed_size_copy_assign<8>::single);
                } else {
                    result->set_function<unary_strided_operation_t>(
                                    unaligned_fixed_size_copy_assign<8>::strided);
                }
                return offset_out + sizeof(ckernel_prefix);
            default:
                // Subtract the base amount to avoid over-reserving memory in this leaf case
                out->ensure_capacity(offset_out + sizeof(unaligned_copy_single_kernel_extra) -
                                sizeof(ckernel_prefix));
                result = out->get_at<ckernel_prefix>(offset_out);
                if (single) {
                    result->set_function<unary_single_operation_t>(&unaligned_copy_single);
                } else {
                    result->set_function<unary_strided_operation_t>(&unaligned_copy_strided);
                }
                reinterpret_cast<unaligned_copy_single_kernel_extra *>(result)->data_size = data_size;
                return offset_out + sizeof(unaligned_copy_single_kernel_extra);
        }
    }
}

static unary_single_operation_t assign_table_single_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            (unary_single_operation_t)&single_assigner_builtin<dst_type, src_type, errmode>::assign
        
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
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, uint8_t), \
        ERROR_MODE_LEVEL(dst_type, uint16_t), \
        ERROR_MODE_LEVEL(dst_type, uint32_t), \
        ERROR_MODE_LEVEL(dst_type, uint64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_uint128), \
        ERROR_MODE_LEVEL(dst_type, dynd_float16), \
        ERROR_MODE_LEVEL(dst_type, float), \
        ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, dynd_float128), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<float>), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<double>) \
    }

    SRC_TYPE_LEVEL(dynd_bool),
    SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),
    SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),
    SRC_TYPE_LEVEL(dynd_int128),
    SRC_TYPE_LEVEL(uint8_t),
    SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),
    SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(dynd_uint128),
    SRC_TYPE_LEVEL(dynd_float16),
    SRC_TYPE_LEVEL(float),
    SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(dynd_float128),
    SRC_TYPE_LEVEL(dynd_complex<float>),
    SRC_TYPE_LEVEL(dynd_complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
#undef SINGLE_OPERATION_PAIR_LEVEL
};

namespace {
    template<typename dst_type, typename src_type, assign_error_mode errmode>
    struct multiple_assignment_builtin {
        static void strided_assign(
                        char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(extra))
        {
            for (size_t i = 0; i != count; ++i, dst += dst_stride, src += src_stride) {
                single_assigner_builtin<dst_type, src_type, errmode>::assign(
                                reinterpret_cast<dst_type *>(dst),
                                reinterpret_cast<const src_type *>(src),
                                NULL);
            }
        }
    };
    template<typename dst_type, typename src_type>
    struct multiple_assignment_builtin<dst_type, src_type, assign_error_none> {
         DYND_CUDA_HOST_DEVICE static void strided_assign(
                        char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(extra))
        {
            for (size_t i = 0; i != count; ++i, dst += dst_stride, src += src_stride) {
                single_assigner_builtin<dst_type, src_type, assign_error_none>::assign(
                                reinterpret_cast<dst_type *>(dst),
                                reinterpret_cast<const src_type *>(src),
                                NULL);
            }
        }
    };
} // anonymous namespace

static unary_strided_operation_t assign_table_strided_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            &multiple_assignment_builtin<dst_type, src_type, errmode>::strided_assign
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_none), \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_overflow), \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_fractional), \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_inexact) \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        ERROR_MODE_LEVEL(dst_type, dynd_bool), \
        ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t), \
        ERROR_MODE_LEVEL(dst_type, int32_t), \
        ERROR_MODE_LEVEL(dst_type, int64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, uint8_t), \
        ERROR_MODE_LEVEL(dst_type, uint16_t), \
        ERROR_MODE_LEVEL(dst_type, uint32_t), \
        ERROR_MODE_LEVEL(dst_type, uint64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, dynd_float16), \
        ERROR_MODE_LEVEL(dst_type, float), \
        ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, dynd_float128), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<float>), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<double>) \
    }

    SRC_TYPE_LEVEL(dynd_bool),
    SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),
    SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),
    SRC_TYPE_LEVEL(dynd_int128),
    SRC_TYPE_LEVEL(uint8_t),
    SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),
    SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(dynd_uint128),
    SRC_TYPE_LEVEL(dynd_float16),
    SRC_TYPE_LEVEL(float),
    SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(dynd_float128),
    SRC_TYPE_LEVEL(dynd_complex<float>),
    SRC_TYPE_LEVEL(dynd_complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
#undef STRIDED_OPERATION_PAIR_LEVEL
};

size_t dynd::make_builtin_type_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                type_id_t dst_type_id, type_id_t src_type_id,
                kernel_request_t kernreq, assign_error_mode errmode)
{
    // Do a table lookup for the built-in range of dynd types
    if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id &&
                    src_type_id >= bool_type_id && src_type_id <= complex_float64_type_id &&
                    errmode != assign_error_default) {
        // No need to reserve more space, the space for a leaf is already there
        ckernel_prefix *result = out->get_at<ckernel_prefix>(offset_out);
        switch (kernreq) {
            case kernel_request_single:
                result->set_function<unary_single_operation_t>(
                                assign_table_single_kernel[dst_type_id-bool_type_id]
                                                [src_type_id-bool_type_id][errmode]);
                break;
            case kernel_request_strided:
                result->set_function<unary_strided_operation_t>(
                                assign_table_strided_kernel[dst_type_id-bool_type_id]
                                                [src_type_id-bool_type_id][errmode]);
                break;
            default: {
                stringstream ss;
                ss << "make_builtin_type_assignment_function: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }   
        }
        return offset_out + sizeof(ckernel_prefix);
    } else {
        stringstream ss;
        ss << "Cannot assign from " << ndt::type(src_type_id) << " to " << ndt::type(dst_type_id);
        throw runtime_error(ss.str());
    }
}

static void wrap_single_as_strided_kernel(char *dst, intptr_t dst_stride,
                const char *src, intptr_t src_stride,
                size_t count, ckernel_prefix *extra)
{
    ckernel_prefix *echild = extra + 1;
    unary_single_operation_t opchild = echild->get_function<unary_single_operation_t>();

    for (size_t i = 0; i != count; ++i,
                    dst += dst_stride, src += src_stride) {
        opchild(dst, src, echild);
    }
}
static void simple_wrapper_kernel_destruct(
                ckernel_prefix *extra)
{
    ckernel_prefix *echild = extra + 1;
    if (echild->destructor) {
        echild->destructor(echild);
    }
}

size_t dynd::make_kernreq_to_single_kernel_adapter(
                ckernel_builder *out_ckb, size_t ckb_offset,
                kernel_request_t kernreq)
{
    switch (kernreq) {
        case kernel_request_single: {
            return ckb_offset;
        }
        case kernel_request_strided: {
            out_ckb->ensure_capacity(ckb_offset + sizeof(ckernel_prefix));
            ckernel_prefix *e = out_ckb->get_at<ckernel_prefix>(ckb_offset);
            e->set_function<unary_strided_operation_t>(&wrap_single_as_strided_kernel);
            e->destructor = &simple_wrapper_kernel_destruct;
            return ckb_offset + sizeof(ckernel_prefix);
        }
        default: {
            stringstream ss;
            ss << "make_kernreq_to_single_kernel_adapter: unrecognized request " << (int)kernreq;
            throw runtime_error(ss.str());
        }
    }
}

void dynd::strided_assign_kernel_extra::single(char *dst, const char *src,
                    ckernel_prefix *extra)
{
    extra_type *e = reinterpret_cast<extra_type *>(extra);
    ckernel_prefix *echild = &(e + 1)->base;
    unary_strided_operation_t opchild = echild->get_function<unary_strided_operation_t>();
    opchild(dst, e->dst_stride, src, e->src_stride, e->size, echild);
}
void dynd::strided_assign_kernel_extra::strided(char *dst, intptr_t dst_stride,
                const char *src, intptr_t src_stride,
                size_t count, ckernel_prefix *extra)
{
    extra_type *e = reinterpret_cast<extra_type *>(extra);
    ckernel_prefix *echild = &(e + 1)->base;
    unary_strided_operation_t opchild = echild->get_function<unary_strided_operation_t>();
    intptr_t inner_size = e->size, inner_dst_stride = e->dst_stride,
                    inner_src_stride = e->src_stride;
    for (size_t i = 0; i != count; ++i,
                    dst += dst_stride, src += src_stride) {
        opchild(dst, inner_dst_stride, src, inner_src_stride, inner_size, echild);
    }
}

void dynd::strided_assign_kernel_extra::destruct(
                ckernel_prefix *extra)
{
    extra_type *e = reinterpret_cast<extra_type *>(extra);
    ckernel_prefix *echild = &(e + 1)->base;
    if (echild->destructor) {
        echild->destructor(echild);
    }
}
