//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/types/base_expr_type.hpp>

namespace dynd {
namespace ndt {

  class DYND_API adapt_type : public base_expr_type {
    ndt::type m_value_tp;
    ndt::type m_storage_tp;
    nd::callable m_forward;
    nd::callable m_inverse;

  public:
    adapt_type(const ndt::type &value_tp, const ndt::type &storage_tp, const nd::callable &forward,
               const nd::callable &inverse);

    adapt_type(const nd::callable &forward, const nd::callable &inverse);

    const ndt::type &get_value_type() const { return m_value_tp; }
    const ndt::type &get_storage_type() const { return m_storage_tp; }
    const nd::callable &get_forward() const { return m_forward; }
    const nd::callable &get_inverse() const { return m_inverse; }

    const type &get_operand_type() const { return m_storage_tp; }

    type with_replaced_storage_type(const type &DYND_UNUSED(replacement_type)) const
    {
      throw std::runtime_error("with_replaced_storage_type is not implemented in adapt_type");
    }

    type get_canonical_type() const { return get_value_type(); }

    bool is_expression() const { return true; }

    void print_type(std::ostream &o) const;
    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    bool operator==(const base_type &rhs) const;
  };

} // namespace dynd::ndt
} // namespace dynd
