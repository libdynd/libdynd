//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__PROPERTY_TYPE_HPP_
#define _DYND__PROPERTY_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/dtype_assign.hpp>

namespace dynd {

class property_type : public base_expression_type {
    ndt::type m_value_type, m_operand_type;
    bool m_readable, m_writable;
    // If this is true, the property is actually on
    // the value_type, and the getters/setters are
    // exchanged.
    bool m_reversed_property;
    std::string m_property_name;
    size_t m_property_index;
public:
    /**
     * Constructs a normal property dtype
     *
     * \param operand_type  The dtype of the operand, which has the given property.
     * \param property_name  The property name.
     * \param property_index  If already known, can be provided to avoid looking up
     *                        the index from the name.
     */
    property_type(const ndt::type& operand_type, const std::string& property_name,
                size_t property_index = std::numeric_limits<size_t>::max());
    /**
     * Constructs a reversed property dtype (property is from value_type
     * instead of operand_type).
     *
     * \param value_type  The dtype of the value, which has the given property.
     * \param operand_type  The dtype of the operand, whose value dtype much match the
     *                       type of the property on value_type.
     * \param property_name  The property name.
     * \param property_index  If already known, can be provided to avoid looking up
     *                        the index from the name.
     */
    property_type(const ndt::type& value_type, const ndt::type& operand_type, const std::string& property_name,
                size_t property_index = std::numeric_limits<size_t>::max());

    virtual ~property_type();

    inline bool is_reversed_property() const {
        return m_reversed_property;
    }

    inline const std::string& get_property_name() const {
        return m_property_name;
    }

    const ndt::type& get_value_type() const {
        return m_value_type;
    }
    const ndt::type& get_operand_type() const {
        return m_operand_type;
    }
    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    void get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *metadata) const;

    bool is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const;

    bool operator==(const base_type& rhs) const;

    ndt::type with_replaced_storage_type(const ndt::type& replacement_type) const;

    size_t make_operand_to_value_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    size_t make_value_to_operand_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    void get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const;
};

namespace ndt {
    /**
     * Makes a property dtype for accessing a named element-wise property
     * of the provided operand dtype.
     */
    inline ndt::type make_property(const ndt::type& operand_type, const std::string& property_name,
                    size_t property_index = std::numeric_limits<size_t>::max()) {
        return ndt::type(new property_type(operand_type, property_name, property_index), false);
    }

    /**
     * Makes a reversed property dtype for viewing the operand as the output
     * of a property of value_type (with its getters/setters exchanged).
     */
    inline ndt::type make_reversed_property(const ndt::type& value_type,
                    const ndt::type& operand_type, const std::string& property_name,
                    size_t property_index = std::numeric_limits<size_t>::max()) {
        return ndt::type(new property_type(value_type, operand_type, property_name, property_index), false);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__PROPERTY_TYPE_HPP_
