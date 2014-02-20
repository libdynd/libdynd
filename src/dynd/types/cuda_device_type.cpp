//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifdef DYND_CUDA

#include <dynd/types/cuda_device_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

cuda_device_type::cuda_device_type(const ndt::type& target_tp)
    : base_memory_type(cuda_device_type_id, memory_kind, target_tp.get_data_size(),
        target_tp.get_data_alignment(), target_tp.get_flags(), target_tp.get_metadata_size(), target_tp.get_ndim(), target_tp)
{
}

cuda_device_type::~cuda_device_type()
{
}

void cuda_device_type::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    m_target_tp.print_data(o, metadata, data);
}

void cuda_device_type::print_type(std::ostream& o) const
{
    o << "cuda_device(" << m_target_tp << ")";
}

void cuda_device_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                const char *metadata, const char *data) const
{
    m_target_tp.extended()->get_shape(ndim, i, out_shape, metadata, data);
}

void cuda_device_type::get_strides(size_t i, intptr_t *out_strides, const char *metadata) const
{
    m_target_tp.extended()->get_strides(i, out_strides, metadata);
}

void cuda_device_type::metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_default_construct(metadata, ndim, shape);
    }
}

void cuda_device_type::metadata_destruct(char *metadata) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_destruct(metadata);
    }
}

void cuda_device_type::data_alloc(char **data, size_t size) const
{
    throw_if_not_cuda_success(cudaMalloc(data, size));
}

void cuda_device_type::data_zeroinit(char *data, size_t size) const
{
    throw_if_not_cuda_success(cudaMemset(data, 0, size));
}

#include <stdio.h>

inline cudaMemcpyKind get_cuda_memcpy_kind(const ndt::type& dst_tp, const ndt::type& src_tp) {
    if (dst_tp.get_type_id() == cuda_device_type_id) {
        if (src_tp.get_type_id() == cuda_device_type_id) {
            return cudaMemcpyDeviceToDevice;
        }

        return cudaMemcpyHostToDevice;
    }

    if (src_tp.get_type_id() == cuda_device_type_id) {
        return cudaMemcpyDeviceToHost;
    }

    return cudaMemcpyHostToHost;
}

inline kernel_request_t get_single_cuda_kernreq(const ndt::type& dst_tp, const ndt::type& src_tp) {
    if (dst_tp.get_type_id() == cuda_device_type_id) {
        if (src_tp.get_type_id() == cuda_device_type_id) {
            return kernel_request_single_cuda_device_to_device;
        }

        return kernel_request_single_cuda_host_to_device;
    }

    if (src_tp.get_type_id() == cuda_device_type_id) {
        return kernel_request_single_cuda_device_to_host;
    }

    return kernel_request_single;
}

size_t cuda_device_type::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t DYND_UNUSED(kernreq), assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    const ndt::type& dst_target_tp = dst_tp.get_canonical_type();
    const ndt::type& src_target_tp = src_tp.get_canonical_type();

    if (dst_target_tp.data_layout_compatible_with(src_target_tp)) {
        return ::make_assignment_kernel(out, offset_out,
                        dst_target_tp, dst_metadata + 0 * dst_tp.get_metadata_size(),
                        src_target_tp, src_metadata,
                        get_single_cuda_kernreq(dst_tp, src_tp), errmode, ectx);
    }

    // the assignment needs a cast first
    // only do this if one of the arrays is a scalar or both arrays on the device

    if (this == dst_tp.extended()) {
        return ::make_assignment_kernel(out, offset_out,
                        dst_target_tp, dst_metadata + 0 * dst_tp.get_metadata_size(),
                        src_target_tp, src_metadata,
                        get_single_cuda_kernreq(dst_tp, src_tp), errmode, ectx);
    }

        return ::make_assignment_kernel(out, offset_out,
                        dst_target_tp, dst_metadata + 0 * dst_tp.get_metadata_size(),
                        src_target_tp, src_metadata,
                        get_single_cuda_kernreq(dst_tp, src_tp), errmode, ectx);
}

/*
        out->ensure_capacity(offset_out + sizeof(single_cuda_device_assign_kernel_extra));
        single_cuda_device_assign_kernel_extra *e = out->get_at<single_cuda_device_assign_kernel_extra>(offset_out);
        e->base.set_function<unary_single_operation_t>(&single_cuda_device_assign_kernel_extra::single);
        e->base.destructor = single_cuda_device_assign_kernel_extra::destruct;
        e->size = dst_target_tp.get_data_size();
        e->kind = get_cuda_memcpy_kind(dst_tp, src_tp);

        return offset_out + sizeof(single_cuda_device_assign_kernel_extra);*/

#endif // DYND_CUDA
