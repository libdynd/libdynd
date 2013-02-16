//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__PROPERTY_DTYPE_HPP_
#define _DYND__PROPERTY_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/kernels/kernel_instance.hpp>

namespace dynd {

class property_dtype : public base_expression_dtype {
    dtype m_value_dtype, m_operand_dtype;
    bool m_readable, m_writable;
    // If this is true, the property is actually on
    // the value_dtype, and the getters/setters are
    // exchanged.
    bool m_reversed_property;
    std::string m_property_name;
    size_t m_property_index;
public:
    /** Constructs a normal property dtype */
    property_dtype(const dtype& operand_dtype, const std::string& property_name);
    /**
     * Constructs a reversed property dtype (property is from value_dtype
     * instead of operand_dtype).
     */
    property_dtype(const dtype& value_dtype, const dtype& operand_dtype, const std::string& property_name);

    virtual ~property_dtype();

    inline bool is_reversed_property() const {
        return m_reversed_property;
    }

    inline const std::string& get_property_name() const {
        return m_property_name;
    }

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
                    assignment_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    const eval::eval_context *ectx) const;

    size_t make_value_to_operand_assignment_kernel(
                    assignment_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    const eval::eval_context *ectx) const;
};

/**
 * Makes a property dtype for accessing a named element-wise property
 * of the provided operand dtype.
 */
inline dtype make_property_dtype(const dtype& operand_dtype, const std::string& property_name) {
    return dtype(new property_dtype(operand_dtype, property_name), false);
}

/**
 * Makes a reversed property dtype for viewing the operand as the output
 * of a property of value_dtype (with its getters/setters exchanged).
 */
inline dtype make_reversed_property_dtype(const dtype& value_dtype,
                const dtype& operand_dtype, const std::string& property_name) {
    return dtype(new property_dtype(value_dtype, operand_dtype, property_name), false);
}

} // namespace dynd

#endif // _DYND__PROPERTY_DTYPE_HPP_
