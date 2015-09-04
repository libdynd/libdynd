//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

namespace dynd {
namespace ndt {

  class DYND_API array_type : public base_expr_type {
    type m_value_tp;

  public:
    array_type(const type &value_tp);

    virtual ~array_type();

    const type &get_operand_type() const
    {
      static type tp;
      return tp;
    }

    const type &get_value_type() const
    {
      return m_value_tp.value_type();
    }

    bool operator==(const base_type &rhs) const;

    virtual void data_construct(const char *arrmeta, char *data) const;

    virtual void data_destruct(const char *DYND_UNUSED(arrmeta),
                               char *DYND_UNUSED(data)) const;

    bool is_expression() const
    {
      return m_value_tp.is_expression();
    }

    void print_data(std::ostream &o, const char *arrmeta,
                    const char *data) const;

    void print_type(std::ostream &o) const;

    type with_replaced_storage_type(const type &replacement_tp) const;

    static type make(const type &value_tp)
    {
      return type(new array_type(value_tp), false);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
