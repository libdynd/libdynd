//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/base_memory_type.hpp>
#include "assignment_kernels.cpp"

#include "../types/dynd_complex.cu"
#include "../types/dynd_float16.cu"
#include "../types/dynd_float128.cu"
#include "../types/dynd_int128.cu"
#include "../types/dynd_uint128.cu"

#ifdef DYND_CUDA

static const ndt::type& get_storage_type(const ndt::type& tp) {
    if (tp.get_kind() == memory_kind) {
        return static_cast<const base_memory_type *>(tp.extended())->get_storage_type();
    } else {
        return tp;
    }
}

size_t dynd::make_cuda_assignment_kernel(
                void *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                const ndt::type& src_tp, const char *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx)
{
    assign_error_mode errmode;
    if (dst_tp.get_type_id() == cuda_device_type_id && src_tp.get_type_id() == cuda_device_type_id) {
        errmode = ectx->cuda_device_errmode;
    } else {
        errmode = ectx->errmode;
    }

    if (get_storage_type(dst_tp).is_builtin()) {
        if (get_storage_type(src_tp).is_builtin()) {
            if (errmode != assign_error_nocheck && is_lossless_assignment(dst_tp, src_tp)) {
                errmode = assign_error_nocheck;
            }

            if (get_storage_type(dst_tp).extended() == get_storage_type(src_tp).extended()) {
                return make_cuda_pod_typed_data_assignment_kernel(ckb, ckb_offset,
                                dst_tp.get_type_id() == cuda_device_type_id,
                                src_tp.get_type_id() == cuda_device_type_id,
                                dst_tp.get_data_size(), dst_tp.get_data_alignment(),
                                kernreq);
            } else {
                return make_cuda_builtin_type_assignment_kernel(ckb, ckb_offset,
                                dst_tp.get_type_id() == cuda_device_type_id,
                                get_storage_type(dst_tp).get_type_id(),
                                src_tp.get_type_id() == cuda_device_type_id,
                                get_storage_type(src_tp).get_type_id(),
                                kernreq, errmode);
            }
        } else {
            return src_tp.extended()->make_assignment_kernel(ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_tp, src_arrmeta,
                            kernreq, ectx);
        }
    } else {
        return dst_tp.extended()->make_assignment_kernel(ckb, ckb_offset,
                        dst_tp, dst_arrmeta,
                        src_tp, src_arrmeta,
                        kernreq, ectx);
    }
}


static expr_single_t assign_table_single_cuda_host_to_device_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            (expr_single_t)&kernels::cuda_host_to_device_assign_ck<dst_type, src_type, errmode>::single_wrapper
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck), \
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

static expr_single_t assign_table_single_cuda_device_to_host_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            (expr_single_t)&kernels::cuda_device_to_host_assign_ck<dst_type, src_type, errmode>::single_wrapper
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck), \
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

static expr_single_t assign_table_single_cuda_device_to_device_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
        &kernels::cuda_device_to_device_assign_ck<dst_type, src_type, errmode>::single_wrapper
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck), \
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

static expr_strided_t assign_table_strided_cuda_device_to_device_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            &kernels::cuda_device_to_device_assign_ck<dst_type, src_type, errmode>::strided_wrapper
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck), \
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


namespace {
template <typename dst_type, typename src_type, assign_error_mode errmode>
struct multiple_assignment_builtin {
  static void strided_assign(char *dst, intptr_t dst_stride, char *const *src,
                             const intptr_t *src_stride, size_t count,
                             ckernel_prefix *DYND_UNUSED(self))
  {
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      single_assigner_builtin<dst_type, src_type, errmode>::assign(
          reinterpret_cast<dst_type *>(dst),
          reinterpret_cast<src_type *>(src0));
      dst += dst_stride;
      src0 += src0_stride;
    }
  }
};
template <typename dst_type, typename src_type>
struct multiple_assignment_builtin<dst_type, src_type, assign_error_nocheck> {
  DYND_CUDA_HOST_DEVICE static void
  strided_assign(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count,
                 ckernel_prefix *DYND_UNUSED(self))
  {
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      single_assigner_builtin<dst_type, src_type, assign_error_nocheck>::assign(
          reinterpret_cast<dst_type *>(dst),
          reinterpret_cast<src_type *>(src0));
      dst += dst_stride;
      src0 += src0_stride;
    }
  }
};
} // anonymous namespace

static expr_strided_t assign_table_strided_kernel[builtin_type_id_count -
                                                  2][builtin_type_id_count -
                                                     2][4] = {
#define STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode)              \
  &kernels::assign_ck<dst_type, src_type, errmode>::strided_wrapper

#define ERROR_MODE_LEVEL(dst_type, src_type)                                   \
  {                                                                            \
    STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck),    \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type,                       \
                                     assign_error_overflow),                   \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type,                       \
                                     assign_error_fractional),                 \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_inexact) \
  }

#define SRC_TYPE_LEVEL(dst_type)                                               \
  {                                                                            \
    ERROR_MODE_LEVEL(dst_type, dynd_bool), ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t),                                   \
        ERROR_MODE_LEVEL(dst_type, int32_t),                                   \
        ERROR_MODE_LEVEL(dst_type, int64_t),                                   \
        ERROR_MODE_LEVEL(dst_type, dynd_int128),                               \
        ERROR_MODE_LEVEL(dst_type, uint8_t),                                   \
        ERROR_MODE_LEVEL(dst_type, uint16_t),                                  \
        ERROR_MODE_LEVEL(dst_type, uint32_t),                                  \
        ERROR_MODE_LEVEL(dst_type, uint64_t),                                  \
        ERROR_MODE_LEVEL(dst_type, dynd_int128),                               \
        ERROR_MODE_LEVEL(dst_type, dynd_float16),                              \
        ERROR_MODE_LEVEL(dst_type, float), ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, dynd_float128),                             \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<float>),                       \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<double>)                       \
  }

    SRC_TYPE_LEVEL(dynd_bool),           SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),             SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),             SRC_TYPE_LEVEL(dynd_int128),
    SRC_TYPE_LEVEL(uint8_t),             SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),            SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(dynd_uint128),        SRC_TYPE_LEVEL(dynd_float16),
    SRC_TYPE_LEVEL(float),               SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(dynd_float128),       SRC_TYPE_LEVEL(dynd_complex<float>),
    SRC_TYPE_LEVEL(dynd_complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
#undef STRIDED_OPERATION_PAIR_LEVEL
};

// This is meant to reflect make_builtin_type_assignment_kernel
size_t dynd::make_cuda_builtin_type_assignment_kernel(
                void *out, intptr_t offset_out,
                bool dst_device, type_id_t dst_type_id,
                bool src_device, type_id_t src_type_id,
                kernel_request_t kernreq, assign_error_mode errmode)
{
    if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id &&
                    src_type_id >= bool_type_id && src_type_id <= complex_float64_type_id &&
                    errmode != assign_error_default) {
        ckernel_prefix *result = reinterpret_cast<ckernel_builder<kernel_request_host> *>(out)->get_at<ckernel_prefix>(offset_out);
        switch (kernreq) {
            case kernel_request_single:
                if (dst_device) {
                    if (src_device) {
                        result->set_function<expr_single_t>(
                                        assign_table_single_cuda_device_to_device_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                        offset_out += sizeof(ckernel_prefix);

                    } else {
                        result->set_function<expr_single_t>(
                                        assign_table_single_cuda_host_to_device_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                        offset_out += sizeof(ckernel_prefix);
                    }
                } else {
                    if (src_device) {
                        result->set_function<expr_single_t>(
                                        assign_table_single_cuda_device_to_host_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                        offset_out += sizeof(ckernel_prefix);
                    } else {
                    (*assign_create[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode])(out, kernreq, offset_out);
                    }
                }
                break;
            case kernel_request_strided:
                if (dst_device) {
                    if (src_device) {
                        result->set_function<expr_strided_t>(
                                        assign_table_strided_cuda_device_to_device_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                        offset_out += sizeof(ckernel_prefix);
                    } else {
                        offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, 1, kernreq);
                        result = reinterpret_cast<ckernel_builder<kernel_request_host> *>(out)->get_at<ckernel_prefix>(offset_out);
                        result->set_function<expr_single_t>(
                                        assign_table_single_cuda_host_to_device_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                        offset_out += sizeof(ckernel_prefix);
                    }
                } else {
                    if (src_device) {
                        offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, 1, kernreq);
                        result = reinterpret_cast<ckernel_builder<kernel_request_host> *>(out)->get_at<ckernel_prefix>(offset_out);
                        result->set_function<expr_single_t>(
                                        assign_table_single_cuda_device_to_host_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                        offset_out += sizeof(ckernel_prefix);
                    } else {
                        result->set_function<expr_strided_t>(
                                        assign_table_strided_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                        offset_out += sizeof(ckernel_prefix);
                    }
                }
                break;
            default: {
                stringstream ss;
                ss << "make_cuda_builtin_type_assignment_function: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }   
        }
        return offset_out;
    } else {
        stringstream ss;
        ss << "Cannot assign from " << ndt::type(src_type_id);
        if (src_device) {
            ss << " in CUDA global memory";
        }
        ss << " to " << ndt::type(dst_type_id);
        if (dst_device) {
            ss << " in CUDA global memory";
        }
        throw runtime_error(ss.str());
    }
}

// This is meant to reflect make_pod_typed_data_assignment_kernel
size_t dynd::make_cuda_pod_typed_data_assignment_kernel(
                void *out, intptr_t offset_out,
                bool dst_device, bool src_device,
                size_t data_size, size_t data_alignment,
                kernel_request_t kernreq)
{
    if (dst_device) {
        if (src_device) {
            kernels::cuda_device_to_device_copy_ck::create(out, kernreq, offset_out, data_size);
            return offset_out;
        } else {
            kernels::cuda_host_to_device_copy_ck::create(out, kernreq, offset_out, data_size);
            return offset_out;
        }
    } else {
        if (src_device) {
            kernels::cuda_device_to_host_copy_ck::create(out, kernreq, offset_out, data_size);
            return offset_out;
        } else {
            return make_pod_typed_data_assignment_kernel(out, offset_out, data_size, data_alignment, kernreq);
        }
    }
}

#endif // DYND_CUDA
