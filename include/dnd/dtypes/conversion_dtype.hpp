//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The conversion dtype represents one dtype viewed
// as another buffering based on the casting mechanism.
//
// This dtype takes on the characteristics of its storage dtype
// through the dtype interface, except for the "kind" which
// is expression_kind to signal that the value_dtype must be examined.
//
#ifndef _DND__CONVERSION_DTYPE_HPP_
#define _DND__CONVERSION_DTYPE_HPP_

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/kernels/unary_kernel_instance.hpp>

namespace dnd {

class conversion_dtype : public extended_dtype {
    dtype m_value_dtype, m_operand_dtype;
    assign_error_mode m_errmode;
    unary_specialization_kernel_instance m_to_value_kernel, m_to_operand_kernel;
public:
    conversion_dtype(const dtype& value_dtype, const dtype& operand_dtype, assign_error_mode errmode);

    type_id_t type_id() const {
        return conversion_type_id;
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

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    bool is_object_type() const {
        return m_operand_dtype.is_object_type();
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const extended_dtype& rhs) const;

    // For expression_kind dtypes - converts to/from the storage's value dtype
    const unary_specialization_kernel_instance& get_operand_to_value_kernel() const;
    const unary_specialization_kernel_instance& get_value_to_operand_kernel() const;
    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;
};

/**
 * Makes a conversion dtype to convert from the operand_dtype to the value_dtype.
 * If the value_dtype has expression_kind, it chains operand_dtype.value_dtype()
 * into value_dtype.storage_dtype().
 */
inline dtype make_conversion_dtype(const dtype& value_dtype, const dtype& operand_dtype, assign_error_mode errmode = default_error_mode) {
    if (operand_dtype.value_dtype() != value_dtype) {
        if (value_dtype.kind() != expression_kind) {
            // Create a conversion dtype when the value kind is different
            return dtype(make_shared<conversion_dtype>(value_dtype, operand_dtype, errmode));
        } else if (value_dtype.storage_dtype() == operand_dtype.value_dtype()) {
            // No conversion required at the connection
            return value_dtype.extended()->with_replaced_storage_dtype(operand_dtype);
        } else {
            // A conversion required at the connection
            return value_dtype.extended()->with_replaced_storage_dtype(
                dtype(make_shared<conversion_dtype>(value_dtype.storage_dtype(), operand_dtype, errmode)));
        }
    } else {
        return operand_dtype;
    }
}

template<typename Tvalue, typename Tstorage>
dtype make_conversion_dtype(assign_error_mode errmode = default_error_mode) {
    return dtype(make_shared<conversion_dtype>(make_dtype<Tvalue>(), make_dtype<Tstorage>(), errmode));
}

} // namespace dnd

#endif // _DND__CONVERSION_DTYPE_HPP_
