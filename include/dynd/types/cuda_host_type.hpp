//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CUDA_HOST_TYPE_HPP_
#define _DYND__CUDA_HOST_TYPE_HPP_

#ifdef DYND_CUDA

#include <cuda_runtime.h>

#include <dynd/type.hpp>
#include <dynd/types/base_memory_type.hpp>
#include <dynd/types/cuda_device_type.hpp>

namespace dynd {

class cuda_host_type : public base_memory_type {
    unsigned int m_cuda_host_flags;
public:
    cuda_host_type(const ndt::type& storage_tp, unsigned int cuda_host_flags = cudaHostAllocDefault);

    virtual ~cuda_host_type();

    inline unsigned int get_cuda_host_flags() const {
        return m_cuda_host_flags;
    }

    void print_type(std::ostream& o) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& storage_tp) const;

    void data_alloc(char **data, size_t size) const;
    void data_zeroinit(char *data, size_t size) const;
    void data_free(char *data) const;

    size_t make_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const ndt::type& dst_tp, const char *dst_metadata,
                    const ndt::type& src_tp, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;
};

namespace ndt {
    inline ndt::type make_cuda_host(const ndt::type& storage_tp, unsigned int cuda_host_flags = cudaHostAllocDefault) {
        return ndt::type(new cuda_host_type(storage_tp, cuda_host_flags), false);
    }
} // namespace ndt

} // namespace dynd

#endif // DYND_CUDA

#endif // _DYND__CUDA_HOST_TYPE_HPP_
