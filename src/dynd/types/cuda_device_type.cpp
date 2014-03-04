//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifdef DYND_CUDA

#include <dynd/types/cuda_device_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

cuda_device_type::cuda_device_type(const ndt::type& storage_tp)
    : base_memory_type(cuda_device_type_id, storage_tp, storage_tp.get_data_size(),
        get_cuda_device_data_alignment(storage_tp), 0, storage_tp.get_flags() | type_flag_not_host_readable)
{
    if (!storage_tp.is_builtin()) {
        throw std::runtime_error("only built-in types may be allocated in CUDA global memory");
    }
}

cuda_device_type::~cuda_device_type()
{
}

void cuda_device_type::print_type(std::ostream& o) const
{
    o << "cuda_device(" << m_storage_tp << ")";
}

bool cuda_device_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != cuda_device_type_id) {
        return false;
    } else {
        const cuda_device_type *dt = static_cast<const cuda_device_type*>(&rhs);
        return m_storage_tp == dt->m_storage_tp;
    }
}

ndt::type cuda_device_type::with_replaced_storage_type(const ndt::type& storage_tp) const
{
    return make_cuda_device(storage_tp);
}

void cuda_device_type::data_alloc(char **data, size_t size) const
{
    throw_if_not_cuda_success(cudaMalloc(data, size));
}

void cuda_device_type::data_zeroinit(char *data, size_t size) const
{
    throw_if_not_cuda_success(cudaMemset(data, 0, size));
}

void cuda_device_type::data_free(char *data) const
{
    throw_if_not_cuda_success(cudaFree(data));
}

size_t cuda_device_type::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    return make_cuda_assignment_kernel(out, offset_out,
                        dst_tp, dst_metadata,
                        src_tp, src_metadata,
                        kernreq, errmode, ectx);
}

#endif // DYND_CUDA
