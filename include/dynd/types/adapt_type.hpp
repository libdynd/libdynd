//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>

namespace dynd {
namespace ndt {

  class DYND_API adapt_type : public base_expr_type {
    type m_value_type, m_operand_type;
    std::string m_op;
    nd::callable m_forward, m_reverse;

  public:
    adapt_type(const type &operand_type, const type &value_type,
               const std::string &op);

    virtual ~adapt_type();

    const type &get_value_type() const { return m_value_type; }
    const type &get_operand_type() const { return m_operand_type; }
    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

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

    static type make(const type &operand_type, const type &value_type,
                     const std::string &op)
    {
      return type(new adapt_type(operand_type, value_type, op), false);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
