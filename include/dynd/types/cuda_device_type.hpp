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
    cuda_device_type(const ndt::type& target_tp);

    virtual ~cuda_device_type();

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_type(std::ostream& o) const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                    const char *metadata, const char *data) const;

    void get_strides(size_t i, intptr_t *out_strides, const char *metadata) const;

    ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const;

    ndt::type at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

    ndt::type get_type_at_dimension(char **inout_metadata, intptr_t i, intptr_t total_ndim = 0) const;

    void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
  //  void metadata_reset_buffers(char *metadata) const;
    //void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
   // void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    void data_alloc(char **data, size_t size) const;
    void data_zeroinit(char *data, size_t size) const;

    size_t make_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const ndt::type& dst_tp, const char *dst_metadata,
                    const ndt::type& src_tp, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;
};

namespace ndt {
    inline ndt::type make_cuda_device(const ndt::type& target_tp) {
        return ndt::type(new cuda_device_type(target_tp), false);
    }
} // namespace ndt

} // namespace dynd

#endif // DYND_CUDA

#endif // _DYND__CUDA_DEVICE_TYPE_HPP_
