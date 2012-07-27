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

#ifndef _DND__POINTER_DTYPE_HPP_
#define _DND__POINTER_DTYPE_HPP_

#include <dnd/dtype.hpp>
#include <dnd/dtypes/void_pointer_dtype.hpp>

namespace dnd {

class pointer_dtype : public extended_dtype {
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
    unsigned char alignment() const {
        return sizeof(void *);
    }
    uintptr_t element_size() const {
        return sizeof(void *);
    }

    const dtype& value_dtype(const dtype& DND_UNUSED(self)) const {
        return m_target_dtype;
    }
    const dtype& operand_dtype(const dtype& DND_UNUSED(self)) const {
        if (m_target_dtype.type_id() == pointer_type_id) {
            return m_target_dtype;
        } else {
            return m_void_pointer_dtype;
        }
    }

    void print_element(std::ostream& o, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return blockref_memory_management;
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;
};

inline dtype make_pointer_dtype(const dtype& target_dtype) {
    if (target_dtype.type_id() != void_type_id) {
        return dtype(make_shared<pointer_dtype>(target_dtype));
    } else {
        return dtype(make_shared<void_pointer_dtype>());
    }
}

template<typename Tnative>
dtype make_pointer_dtype() {
    return make_pointer_dtype(make_dtype<Tnative>());
}

} // namespace dnd

#endif // _DND__POINTER_DTYPE_HPP_
