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
                        sizeof(void *), dtype_flag_scalar|dtype_flag_zeroinit, 0)
    {}

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return blockref_memory_management;
    }

    void get_shape(size_t i, intptr_t *out_shape) const;
    void get_shape(size_t i, intptr_t *out_shape, const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(kernel_instance<compare_operations_t>& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const;

    bool operator==(const base_dtype& rhs) const;

    size_t get_metadata_size() const {
        return 0;
    }
    void metadata_default_construct(char *DYND_UNUSED(metadata), int DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const {
    }
    void metadata_copy_construct(char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const {
    }
    void metadata_destruct(char *DYND_UNUSED(metadata)) const {
    }
    void metadata_debug_print(const char *DYND_UNUSED(metadata), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }
};

} // namespace dynd

#endif // _DYND__VOID_POINTER_DTYPE_HPP_
