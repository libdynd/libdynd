//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The view type reinterprets the bytes of
// one type as another.
//

#pragma once

#include <dynd/type.hpp>

namespace dynd {
namespace ndt {

  class DYND_API view_type : public base_expr_type {
    type m_value_type, m_operand_type;

  public:
    view_type(const type &value_type, const type &operand_type);

    virtual ~view_type();

    const type &get_value_type() const { return m_value_type; }
    const type &get_operand_type() const { return m_operand_type; }
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

    // Propagate properties and functions from the value type
    void get_dynamic_array_properties(
        const std::pair<std::string, gfunc::callable> **out_properties,
        size_t *out_count) const
    {
      if (!m_value_type.is_builtin()) {
        m_value_type.extended()->get_dynamic_array_properties(out_properties,
                                                              out_count);
      }
    }
    void get_dynamic_array_functions(
        const std::pair<std::string, gfunc::callable> **out_functions,
        size_t *out_count) const
    {
      if (!m_value_type.is_builtin()) {
        m_value_type.extended()->get_dynamic_array_functions(out_functions,
                                                             out_count);
      }
    }

    /**
     * Makes an unaligned type to view the given type without alignment
     * requirements.
     */
    static type make(const type &value_type, const type &operand_type)
    {
      if (value_type.get_kind() != expr_kind) {
        return type(new view_type(value_type, operand_type), false);
      } else {
        // When the value type has an expr_kind, we need to chain things
        // together
        // so that the view operation happens just at the primitive level.
        return value_type.extended<base_expr_type>()
            ->with_replaced_storage_type(type(
                new view_type(value_type.storage_type(), operand_type), false));
      }
    }
  };

} // namespace dynd::ndt
} // namespace dynd
