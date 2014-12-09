//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_memory_type.hpp>

namespace dynd {

#ifdef DYND_CUDA

class cuda_device_type : public base_memory_type {
public:
    cuda_device_type(const ndt::type& element_tp);

    virtual ~cuda_device_type();

    void print_type(std::ostream& o) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& element_tp) const;

    void data_alloc(char **data, size_t size) const;
    void data_zeroinit(char *data, size_t size) const;
    void data_free(char *data) const;

    intptr_t make_assignment_kernel(
        const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        const ndt::type &src_tp, const char *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx,
        const nd::array &kwds) const;
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
    inline ndt::type make_cuda_device(const ndt::type& element_tp) {
        return ndt::type(new cuda_device_type(element_tp), false);
    }
} // namespace ndt

#endif

} // namespace dynd