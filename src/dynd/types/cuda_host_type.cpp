//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/config.hpp>

#ifdef DYND_CUDA

#include <dynd/types/cuda_host_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

cuda_host_type::cuda_host_type(const ndt::type& storage_tp, unsigned int cuda_host_flags)
    : base_memory_type(cuda_host_type_id, storage_tp, storage_tp.get_data_size(),
        get_cuda_device_data_alignment(storage_tp), 0, storage_tp.get_flags()),
        m_cuda_host_flags(cuda_host_flags)
{
}

cuda_host_type::~cuda_host_type()
{
}

void cuda_host_type::print_type(std::ostream& o) const
{
    o << "cuda_host[" << m_storage_tp << "]";
}

bool cuda_host_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != cuda_host_type_id) {
        return false;
    } else {
        const cuda_host_type *dt = static_cast<const cuda_host_type*>(&rhs);
        return m_storage_tp == dt->m_storage_tp && m_cuda_host_flags == dt->get_cuda_host_flags();
    }
}

ndt::type cuda_host_type::with_replaced_storage_type(const ndt::type& storage_tp) const
{
    return make_cuda_host(storage_tp, m_cuda_host_flags);
}

void cuda_host_type::data_alloc(char **data, size_t size) const
{
    throw_if_not_cuda_success(cudaHostAlloc(data, size, m_cuda_host_flags));
}

void cuda_host_type::data_zeroinit(char *data, size_t size) const
{
    memset(data, 0, size);
}

void cuda_host_type::data_free(char *data) const
{
    throw_if_not_cuda_success(cudaFreeHost(data));
}

size_t cuda_host_type::make_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    return make_cuda_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                       src_tp, src_arrmeta, kernreq, ectx);
}

#endif // DYND_CUDA
