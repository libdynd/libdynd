//
// Copyright (C) 2011-12, Dynamic NDArray Developers
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

class void_pointer_dtype : public extended_dtype {
public:
    void_pointer_dtype() {
    }

    type_id_t type_id() const {
        return void_pointer_type_id;
    }
    dtype_kind_t kind() const {
        return void_kind;
    }
    // Expose the storage traits here
    size_t alignment() const {
        return sizeof(void *);
    }
    uintptr_t element_size() const {
        return sizeof(void *);
    }

    void print_element(std::ostream& o, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return blockref_memory_management;
    }

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    void get_shape(int i, std::vector<intptr_t>& out_shape) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;
};

} // namespace dynd

#endif // _DYND__VOID_POINTER_DTYPE_HPP_
