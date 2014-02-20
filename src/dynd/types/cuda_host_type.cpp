//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifdef DYND_CUDA

#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

cuda_host_type::cuda_host_type(const ndt::type& target_tp, unsigned int cuda_host_flags)
    : base_memory_type(cuda_host_type_id, memory_kind, target_tp.get_data_size(),
        target_tp.get_data_alignment(), target_tp.get_flags(), target_tp.get_metadata_size(),
        target_tp.get_ndim(), target_tp), m_cuda_host_flags(cuda_host_flags)
{
}

cuda_host_type::~cuda_host_type()
{
}

void cuda_host_type::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    m_target_tp.print_data(o, metadata, data);
}

void cuda_host_type::print_type(std::ostream& o) const
{
    o << "cuda_host(" << m_target_tp << ")";
}

void cuda_host_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                const char *metadata, const char *data) const
{
    m_target_tp.extended()->get_shape(ndim, i, out_shape, metadata, data);
}

void cuda_host_type::get_strides(size_t i, intptr_t *out_strides, const char *metadata) const
{
    m_target_tp.extended()->get_strides(i, out_strides, metadata);
}

void cuda_host_type::metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_default_construct(metadata, ndim, shape);
    }
}

void cuda_host_type::metadata_destruct(char *metadata) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_destruct(metadata);
    }
}

void cuda_host_type::data_alloc(char **data, size_t size) const
{
    throw_if_not_cuda_success(cudaHostAlloc(data, size, m_cuda_host_flags));
}

void cuda_host_type::data_zeroinit(char *data, size_t size) const
{
    memset(data, 0, size);
}

size_t cuda_host_type::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        if (src_tp.get_type_id() == cuda_device_type_id) {
            return reinterpret_cast<const cuda_device_type*>(src_tp.extended())->make_assignment_kernel(out,
                offset_out, dst_tp, dst_metadata, src_tp, src_metadata, kernreq, errmode, ectx);
        }

        return ::make_assignment_kernel(out, offset_out, m_target_tp, dst_metadata + get_metadata_size(),
            src_tp, src_metadata, kernreq, errmode, ectx);
    }

    return ::make_assignment_kernel(out, offset_out, dst_tp, dst_metadata,
        m_target_tp, src_metadata + get_metadata_size(), kernreq, errmode, ectx);
}

#endif // DYND_CUDA
