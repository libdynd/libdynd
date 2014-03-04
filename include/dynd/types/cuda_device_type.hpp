//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CUDA_DEVICE_TYPE_HPP_
#define _DYND__CUDA_DEVICE_TYPE_HPP_

#ifdef DYND_CUDA

#include <dynd/type.hpp>
#include <dynd/types/base_memory_type.hpp>

namespace dynd {

class cuda_device_type : public base_memory_type {
public:
    cuda_device_type(const ndt::type& storage_tp);

    virtual ~cuda_device_type();

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

inline size_t get_cuda_device_data_alignment(const ndt::type& dtp) {
    if (dtp.is_builtin()) {
        return dtp.get_data_size();
    } else {
        // TODO: Return the data size of the largest built-in component
        throw std::runtime_error("only built-in types may be allocated in CUDA global memory");
    }
}

namespace ndt {
    inline ndt::type make_cuda_device(const ndt::type& storage_tp) {
        return ndt::type(new cuda_device_type(storage_tp), false);
    }
} // namespace ndt

} // namespace dynd

#endif // DYND_CUDA

#endif // _DYND__CUDA_DEVICE_TYPE_HPP_
