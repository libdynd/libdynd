//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

/**
 * The pointer dtype contains C/C++ raw pointers
 * pointing at data in other memory_blocks, using
 * blockrefs to manage the memory.
 *
 * This dtype operates in a "gather/scatter" fashion,
 * exposing itself as an expression dtype whose expression
 * copies the data to/from the pointer targets.
 */

#ifndef _DYND__POINTER_DTYPE_HPP_
#define _DYND__POINTER_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtypes/void_pointer_dtype.hpp>

namespace dynd {

class pointer_dtype : public extended_expression_dtype {
    dtype m_target_dtype;
    static dtype m_void_pointer_dtype;

public:
    pointer_dtype(const dtype& target_dtype);

    type_id_t type_id() const {
        return pointer_type_id;
    }
    dtype_kind_t kind() const {
        return expression_kind;
    }
    // Expose the storage traits here
    size_t alignment() const {
        return sizeof(void *);
    }
    uintptr_t element_size() const {
        return sizeof(void *);
    }

    const dtype& get_value_dtype() const {
        return m_target_dtype;
    }
    const dtype& get_operand_dtype() const {
        if (m_target_dtype.type_id() == pointer_type_id) {
            return m_target_dtype;
        } else {
            return m_void_pointer_dtype;
        }
    }

    void print_element(std::ostream& o, const char *data, const char *metadata) const;

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

    bool operator==(const extended_dtype& rhs) const;

    // Converts to/from the storage's value dtype
    void get_operand_to_value_kernel(const eval::eval_context *ectx,
                            unary_specialization_kernel_instance& out_borrowed_kernel) const;
    void get_value_to_operand_kernel(const eval::eval_context *ectx,
                            unary_specialization_kernel_instance& out_borrowed_kernel) const;
    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;
};

inline dtype make_pointer_dtype(const dtype& target_dtype) {
    if (target_dtype.type_id() != void_type_id) {
        return dtype(new pointer_dtype(target_dtype));
    } else {
        return dtype(new void_pointer_dtype());
    }
}

template<typename Tnative>
dtype make_pointer_dtype() {
    return make_pointer_dtype(make_dtype<Tnative>());
}

} // namespace dynd

#endif // _DYND__POINTER_DTYPE_HPP_
