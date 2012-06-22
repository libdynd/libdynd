//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The view dtype reinterprets the bytes of
// one dtype as another.
//
#ifndef _DND__VIEW_DTYPE_HPP_
#define _DND__VIEW_DTYPE_HPP_

#include <dnd/dtype.hpp>

namespace dnd {

class view_dtype : public extended_dtype {
    dtype m_value_dtype, m_operand_dtype;
    unary_specialization_kernel_instance m_copy_kernel;

public:
    view_dtype(const dtype& value_dtype, const dtype& operand_dtype);

    type_id_t type_id() const {
        return view_type_id;
    }
    dtype_kind_t kind() const {
        return expression_kind;
    }
    // Expose the storage traits here
    unsigned char alignment() const {
        return m_operand_dtype.alignment();
    }
    uintptr_t element_size() const {
        return m_operand_dtype.element_size();
    }

    const dtype& value_dtype(const dtype& DND_UNUSED(self)) const {
        return m_value_dtype;
    }
    const dtype& operand_dtype(const dtype& DND_UNUSED(self)) const {
        return m_operand_dtype;
    }
    void print_element(std::ostream& o, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // Don't support unaligned versions of object-semantic data
    bool is_object_type() const {
        return false;
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const extended_dtype& rhs) const;

    // For expression_kind dtypes - converts to/from the storage's value dtype
    const unary_specialization_kernel_instance& get_operand_to_value_kernel() const;
    const unary_specialization_kernel_instance& get_value_to_operand_kernel() const;
    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;
};

/**
 * Makes an unaligned dtype to view the given dtype without alignment requirements.
 */
inline dtype make_view_dtype(const dtype& value_dtype, const dtype& operand_dtype) {
    if (value_dtype.kind() != expression_kind) {
        return dtype(make_shared<view_dtype>(value_dtype, operand_dtype));
    } else {
        // When the value dtype has an expression_kind, we need to chain things together
        // so that the view operation happens just at the primitive level.
        return value_dtype.extended()->with_replaced_storage_dtype(
            dtype(make_shared<view_dtype>(value_dtype.storage_dtype(), operand_dtype)));
    }
}

template<typename Tvalue, typename Toperand>
dtype make_view_dtype() {
    return dtype(make_shared<view_dtype>(make_dtype<Tvalue>()));
}

} // namespace dnd

#endif // _DND__VIEW_DTYPE_HPP_
