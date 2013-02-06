//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This is being initially written as a "date" property type, but
// with an appropriate base_dtype interface which generates the
// to_value_kernel, this can be switched to just a property type using
// that interface.

#ifndef _DYND__DATE_PROPERTY_DTYPE_HPP_
#define _DYND__DATE_PROPERTY_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/kernels/kernel_instance.hpp>

namespace dynd {

class date_property_dtype : public base_expression_dtype {
    dtype m_value_dtype, m_operand_dtype;
    std::string m_property_name;
    size_t m_property_index;
public:
    date_property_dtype(const dtype& operand_dtype, const std::string& property_name);

    virtual ~date_property_dtype();

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
inline dtype make_date_property_dtype(const dtype& operand_dtype, const std::string& property_name) {
    return dtype(new date_property_dtype(operand_dtype, property_name), false);
}

} // namespace dynd

#endif // _DYND__DATE_PROPERTY_DTYPE_HPP_
