//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/base_memory_type.hpp>
#include "assignment_kernels.cpp"
#include <dynd/kernels/cuda_kernels.hpp>

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
                const arrfunc_type_data *self, const arrfunc_type *af_tp,
                void *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                const ndt::type& src_tp, const char *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds)
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
            return src_tp.extended()->make_assignment_kernel(self, af_tp, ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_tp, src_arrmeta,
                            kernreq, ectx, kwds);
        }
    } else {
        return dst_tp.extended()->make_assignment_kernel(self, af_tp, ckb, ckb_offset,
                        dst_tp, dst_arrmeta,
                        src_tp, src_arrmeta,
                        kernreq, ectx, kwds);
    }
}


static kernels::create_t assign_table_single_cuda_host_to_device_kernel[builtin_type_id_count-2] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type) \
            &kernels::cuda_host_to_device_assign_ck<sizeof(dst_type)>::create_opaque

    SINGLE_OPERATION_PAIR_LEVEL(dynd_bool),
    SINGLE_OPERATION_PAIR_LEVEL(int8_t),
    SINGLE_OPERATION_PAIR_LEVEL(int16_t),
    SINGLE_OPERATION_PAIR_LEVEL(int32_t),
    SINGLE_OPERATION_PAIR_LEVEL(int64_t),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_int128),
    SINGLE_OPERATION_PAIR_LEVEL(uint8_t),
    SINGLE_OPERATION_PAIR_LEVEL(uint16_t),
    SINGLE_OPERATION_PAIR_LEVEL(uint32_t),
    SINGLE_OPERATION_PAIR_LEVEL(uint64_t),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_uint128),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_float16),
    SINGLE_OPERATION_PAIR_LEVEL(float),
    SINGLE_OPERATION_PAIR_LEVEL(double),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_float128),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_complex<float>),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_complex<double>)

#undef SINGLE_OPERATION_PAIR_LEVEL
};

static kernels::create_t assign_table_single_cuda_device_to_host_kernel[builtin_type_id_count-2] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(src_type) \
            &kernels::cuda_device_to_host_assign_ck<sizeof(src_type)>::create_opaque

    SINGLE_OPERATION_PAIR_LEVEL(dynd_bool),
    SINGLE_OPERATION_PAIR_LEVEL(int8_t),
    SINGLE_OPERATION_PAIR_LEVEL(int16_t),
    SINGLE_OPERATION_PAIR_LEVEL(int32_t),
    SINGLE_OPERATION_PAIR_LEVEL(int64_t),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_int128),
    SINGLE_OPERATION_PAIR_LEVEL(uint8_t),
    SINGLE_OPERATION_PAIR_LEVEL(uint16_t),
    SINGLE_OPERATION_PAIR_LEVEL(uint32_t),
    SINGLE_OPERATION_PAIR_LEVEL(uint64_t),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_uint128),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_float16),
    SINGLE_OPERATION_PAIR_LEVEL(float),
    SINGLE_OPERATION_PAIR_LEVEL(double),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_float128),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_complex<float>),
    SINGLE_OPERATION_PAIR_LEVEL(dynd_complex<double>)

#undef SINGLE_OPERATION_PAIR_LEVEL
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
        switch (kernreq) {
            case kernel_request_single:
                if (dst_device) {
                    if (src_device) {
                        kernels::cuda_parallel_ck<1> *self = kernels::cuda_parallel_ck<1>::create(out, kernreq, offset_out, 1, 1);
                        offset_out = 0;
                        (*assign_create[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode])(self->get_ckb(),
                            kernel_request_cuda_device | kernreq, offset_out);
                    } else {
                        (*assign_table_single_cuda_host_to_device_kernel[dst_type_id-bool_type_id])(out, kernreq, offset_out);
                        (*assign_create[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode])(out, kernel_request_single, offset_out);
                    }
                } else {
                    if (src_device) {
                        (*assign_table_single_cuda_device_to_host_kernel[src_type_id-bool_type_id])(out, kernreq, offset_out);
                        (*assign_create[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode])(out, kernel_request_single, offset_out);
                    } else {
                    (*assign_create[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode])(out, kernreq, offset_out);
                    }
                }
                break;
            case kernel_request_strided:
                if (dst_device) {
                    if (src_device) {
                        kernels::cuda_parallel_ck<1> *self = kernels::cuda_parallel_ck<1>::create(out, kernreq, offset_out, 1, 1);
                        offset_out = 0;
                        (*assign_create[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode])(self->get_ckb(),
                            kernel_request_cuda_device | kernreq, offset_out);
                    } else {
                        (*assign_table_single_cuda_host_to_device_kernel[dst_type_id-bool_type_id])(out, kernreq, offset_out);
                        (*assign_create[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode])(out, kernel_request_single, offset_out);
                    }
                } else {
                    if (src_device) {
                        (*assign_table_single_cuda_device_to_host_kernel[src_type_id-bool_type_id])(out, kernreq, offset_out);
                        (*assign_create[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode])(out, kernel_request_single, offset_out);
                    } else {
                    (*assign_create[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode])(out, kernreq, offset_out);
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
