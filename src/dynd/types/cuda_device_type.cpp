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

void cuda_device_type::print_type(std::ostream& o) const
{
    o << "cuda_device(" << m_target_tp << ")";
}

ndt::type cuda_device_type::with_replaced_target_type(const ndt::type& target_tp) const
{
    return make_cuda_device(target_tp);
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
//    if (dst_tp.is_pod() && src_tp.is_pod() && dst_tp.get_canonical_type() == src_tp.get_canonical_type()) {
    if (false) {
        return make_pod_typed_data_assignment_kernel(out, offset_out, dst_tp.get_data_size(),
            dst_tp.get_data_alignment(), make_kernreq_to_cuda_kernreq(dst_tp, src_tp, kernreq));
    } else if (this == dst_tp.extended()) {
        const char *shifted_metadata = dst_metadata;
        ndt::type shifted_tp = dst_tp.with_shifted_memory_type();
        return ::make_assignment_kernel(out, offset_out,
            shifted_tp, shifted_metadata, src_tp, src_metadata,
            kernreq, errmode, ectx);
    } else if (this == src_tp.extended()) {
        const char *shifted_metadata = src_metadata;
        ndt::type shifted_tp = src_tp.with_shifted_memory_type();
        return ::make_assignment_kernel(out, offset_out,
                dst_tp, dst_metadata, shifted_tp, shifted_metadata,
                kernreq, errmode, ectx);
    }

    cout << "error" << endl;
    return 0;
}

#endif // DYND_CUDA
