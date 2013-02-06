//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The conversion dtype represents one dtype viewed
// as another buffering based on the casting mechanism.
//
// This dtype takes on the characteristics of its storage dtype
// through the dtype interface, except for the "kind" which
// is expression_kind to signal that the value_dtype must be examined.
//
#ifndef _DYND__CONVERT_DTYPE_HPP_
#define _DYND__CONVERT_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/kernels/kernel_instance.hpp>

namespace dynd {

class convert_dtype : public base_expression_dtype {
    dtype m_value_dtype, m_operand_dtype;
    assign_error_mode m_errmode;
    // These error modes may be set to assign_error_none if the
    // assignment is lossless in that direction.
    assign_error_mode m_errmode_to_value, m_errmode_to_operand;
public:
    convert_dtype(const dtype& value_dtype, const dtype& operand_dtype, assign_error_mode errmode);

    virtual ~convert_dtype();

    const dtype& get_value_dtype() const {
        return m_value_dtype;
    }
    const dtype& get_operand_dtype() const {
        return m_operand_dtype;
    }
    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return m_operand_dtype.get_memory_management();
    }

    void get_shape(size_t i, intptr_t *out_shape) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;

    size_t make_operand_to_value_assignment_kernel(
                    hierarchical_kernel<unary_single_operation_t> *out,
                    size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
                    hierarchical_kernel<unary_single_operation_t> *out,
                    size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    const eval::eval_context *ectx) const;

};

/**
 * Makes a conversion dtype to convert from the operand_dtype to the value_dtype.
 * If the value_dtype has expression_kind, it chains operand_dtype.value_dtype()
 * into value_dtype.storage_dtype().
 */
inline dtype make_convert_dtype(const dtype& value_dtype, const dtype& operand_dtype, assign_error_mode errmode = assign_error_default) {
    if (operand_dtype.value_dtype() != value_dtype) {
        if (value_dtype.get_kind() != expression_kind) {
            // Create a conversion dtype when the value kind is different
            return dtype(new convert_dtype(value_dtype, operand_dtype, errmode), false);
        } else if (value_dtype.storage_dtype() == operand_dtype.value_dtype()) {
            // No conversion required at the connection
            return static_cast<const base_expression_dtype *>(value_dtype.extended())->with_replaced_storage_dtype(operand_dtype);
        } else {
            // A conversion required at the connection
            return static_cast<const base_expression_dtype *>(value_dtype.extended())->with_replaced_storage_dtype(
                dtype(new convert_dtype(value_dtype.storage_dtype(), operand_dtype, errmode), false));
        }
    } else {
        return operand_dtype;
    }
}

template<typename Tvalue, typename Tstorage>
dtype make_convert_dtype(assign_error_mode errmode = assign_error_default) {
    return dtype(new convert_dtype(make_dtype<Tvalue>(), make_dtype<Tstorage>(), errmode), false);
}

} // namespace dynd

#endif // _DYND__CONVERT_DTYPE_HPP_
