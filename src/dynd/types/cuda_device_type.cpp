//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifdef DYND_CUDA

#include <dynd/types/cuda_device_type.hpp>

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

#endif // DYND_CUDA
