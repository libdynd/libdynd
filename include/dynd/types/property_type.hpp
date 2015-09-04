//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {
namespace ndt {

  class DYND_API property_type : public base_expr_type {
    type m_value_tp, m_operand_tp;
    bool m_readable, m_writable;
    // If this is true, the property is actually on
    // the value type, and the getters/setters are
    // exchanged.
    bool m_reversed_property;
    std::string m_property_name;
    size_t m_property_index;

  public:
    /**
     * Constructs a normal property type
     *
     * \param operand_tp  The type of the operand, which has the given property.
     * \param property_name  The property name.
     * \param property_index  If already known, can be provided to avoid looking
     *up
     *                        the index from the name.
     */
    property_type(const type &operand_tp, const std::string &property_name,
                  size_t property_index = std::numeric_limits<size_t>::max());
    /**
     * Constructs a reversed property type (property is from value_tp
     * instead of operand_tp).
     *
     * \param value_tp  The type of the value, which has the given property.
     * \param operand_tp  The type of the operand, whose value type much match
     *the
     *                       type of the property on value_tp.
     * \param property_name  The property name.
     * \param property_index  If already known, can be provided to avoid looking
     *up
     *                        the index from the name.
     */
    property_type(const type &value_tp, const type &operand_tp,
                  const std::string &property_name,
                  size_t property_index = std::numeric_limits<size_t>::max());

    virtual ~property_type();

    inline bool is_reversed_property() const { return m_reversed_property; }

    inline const std::string &get_property_name() const
    {
      return m_property_name;
    }

    const type &get_value_type() const { return m_value_tp; }
    const type &get_operand_type() const { return m_operand_tp; }
    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                   const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    type with_replaced_storage_type(const type &replacement_type) const;

    size_t make_operand_to_value_assignment_kernel(
        void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
        const char *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *ectx) const;

    size_t make_value_to_operand_assignment_kernel(
        void *ckb, intptr_t ckb_offset, const char *dst_arrmeta,
        const char *src_arrmeta, kernel_request_t kernreq,
        const eval::eval_context *ectx) const;

    void get_dynamic_array_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const;
    void get_dynamic_array_functions(
        const std::pair<std::string, gfunc::callable> **out_functions,
        size_t *out_count) const;

    /**
     * Makes a property type for accessing a named element-wise property
     * of the provided operand type.
     */
    static type make(const type &operand_tp, const std::string &property_name,
                     size_t property_index = std::numeric_limits<size_t>::max())
    {
      return type(new property_type(operand_tp, property_name, property_index),
                  false);
    }

    /**
     * Makes a reversed property type for viewing the operand as the output
     * of a property of value_tp (with its getters/setters exchanged).
     */
    static type
    make_reversed(const type &value_tp, const type &operand_tp,
                  const std::string &property_name,
                  size_t property_index = std::numeric_limits<size_t>::max())
    {
      return type(new property_type(value_tp, operand_tp, property_name,
                                    property_index),
                  false);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
