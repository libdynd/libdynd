//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The array dtype uses memory_block references to store
// arbitrarily sized arrays.
//
#ifndef _DND__ARRAY_DTYPE_HPP_
#define _DND__ARRAY_DTYPE_HPP_

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/dtypes/view_dtype.hpp>

namespace dnd {

class array_dtype : public extended_dtype {
    dtype m_element_dtype;
public:
    array_dtype(const dtype& element_dtype);

    type_id_t type_id() const {
        return array_type_id;
    }
    dtype_kind_t kind() const {
        return composite_kind;
    }
    // Expose the storage traits here
    size_t alignment() const {
        return sizeof(const char *);
    }
    uintptr_t element_size() const {
        return 2 * sizeof(const char *);
    }

    const dtype& value_dtype(const dtype& self) const {
        return self;
    }
    const dtype& operand_dtype(const dtype& self) const {
        return self;
    }

    const dtype& get_element_dtype() const {
        return m_element_dtype;
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

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;
};

inline dtype make_array_dtype(const dtype& element_dtype) {
    return dtype(make_shared<array_dtype>(element_dtype));
}

} // namespace dnd

#endif // _DND__ARRAY_DTYPE_HPP_
