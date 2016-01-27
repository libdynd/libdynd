//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The conversion type represents one type viewed
// as another buffering based on the casting mechanism.
//
// This type takes on the characteristics of its storage type
// through the type interface, except for the "kind" which
// is expr_kind to signal that the value_type must be examined.
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace ndt {

  class DYND_API convert_type : public base_expr_type {
    type m_value_type, m_operand_type;

  public:
    convert_type(const type &value_type, const type &operand_type);

    virtual ~convert_type();

    const type &get_value_type() const { return m_value_type; }
    const type &get_operand_type() const { return m_operand_type; }
    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    type with_replaced_storage_type(const type &replacement_type) const;

    void make_operand_to_value_assignment_kernel(nd::kernel_builder *ckb, const char *dst_arrmeta,
                                                 const char *src_arrmeta, kernel_request_t kernreq,
                                                 const eval::eval_context *ectx) const;
    void make_value_to_operand_assignment_kernel(nd::kernel_builder *ckb, const char *dst_arrmeta,
                                                 const char *src_arrmeta, kernel_request_t kernreq,
                                                 const eval::eval_context *ectx) const;

    // Propagate properties and functions from the value type
    std::map<std::string, nd::callable> get_dynamic_array_properties() const
    {
      return m_value_type.get_array_properties();
    }

    std::map<std::string, nd::callable> get_dynamic_array_functions() const
    {
      return m_value_type.get_array_functions();
    }

    /**
     * Makes a conversion type to convert from the operand_type to the
     * value_type.
     * If the value_type has expr_kind, it chains operand_type.value_type()
     * into value_type.storage_type().
     */
    static type make(const type &value_type, const type &operand_type)
    {
      if (operand_type.value_type() != value_type) {
        if (value_type.get_base_id() != expr_kind_id) {
          // Create a conversion type when the value kind is different
          return type(new convert_type(value_type, operand_type), false);
        }
        else if (value_type.storage_type() == operand_type.value_type()) {
          // No conversion required at the connection
          return static_cast<const base_expr_type *>(value_type.extended())->with_replaced_storage_type(operand_type);
        }
        else {
          // A conversion required at the connection
          return static_cast<const base_expr_type *>(value_type.extended())
              ->with_replaced_storage_type(type(new convert_type(value_type.storage_type(), operand_type), false));
        }
      }
      else {
        return operand_type;
      }
    }
  };

} // namespace dynd::ndt
} // namespace dynd
