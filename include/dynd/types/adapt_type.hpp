//
// Copyright (C) 2011-16 DyND Developers
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
    adapt_type(type_id_t new_id, const ndt::type &value_tp, const ndt::type &storage_tp, const nd::callable &forward,
               const nd::callable &inverse)
        : base_expr_type(new_id, adapt_id, storage_tp.get_data_size(), storage_tp.get_data_alignment(), type_flag_none,
                         storage_tp.get_arrmeta_size(), storage_tp.get_ndim()),
          m_value_tp(value_tp), m_storage_tp(storage_tp), m_forward(forward), m_inverse(inverse) {}

    adapt_type(type_id_t new_id, const nd::callable &forward, const nd::callable &inverse)
        : adapt_type(new_id, forward->get_ret_type(), forward->get_arg_types()[0], forward, inverse) {}

    const ndt::type &get_value_type() const { return m_value_tp; }
    const ndt::type &get_storage_type() const { return m_storage_tp; }
    const nd::callable &get_forward() const { return m_forward; }
    const nd::callable &get_inverse() const { return m_inverse; }

    const type &get_operand_type() const { return m_storage_tp; }

    type with_replaced_storage_type(const type &DYND_UNUSED(replacement_type)) const {
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
