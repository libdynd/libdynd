//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/**
 * The void pointer dtype serves as the storage for a
 * pointer dtype, breaking the chaining of pointers
 * as expression dtypes.
 */

#ifndef _DYND__VOID_POINTER_DTYPE_HPP_
#define _DYND__VOID_POINTER_DTYPE_HPP_

#include <dynd/dtype.hpp>

namespace dynd {

class void_pointer_dtype : public base_dtype {
public:
    void_pointer_dtype()
        : base_dtype(void_pointer_type_id, void_kind, sizeof(void *),
                        sizeof(void *), dtype_flag_scalar|dtype_flag_zeroinit|dtype_flag_blockref,
                        0, 0)
    {}

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    void get_shape(size_t i, intptr_t *out_shape) const;
    void get_shape(size_t i, intptr_t *out_shape, const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    void metadata_default_construct(char *DYND_UNUSED(metadata), size_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const {
    }
    void metadata_copy_construct(char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const {
    }
    void metadata_destruct(char *DYND_UNUSED(metadata)) const {
    }
    void metadata_debug_print(const char *DYND_UNUSED(metadata), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }

    size_t make_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const dtype& dst_dt, const char *dst_metadata,
                    const dtype& src_dt, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;
};

} // namespace dynd

#endif // _DYND__VOID_POINTER_DTYPE_HPP_
