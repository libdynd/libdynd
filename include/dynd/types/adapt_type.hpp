//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

namespace dynd {
namespace ndt {

  class DYND_API adapt_type : public base_expr_type {
  public:
    adapt_type(const type &operand_type, const type &value_type, const std::string &op);

    const type &get_value_type() const { throw std::runtime_error("not implemented"); }
    const type &get_operand_type() const { throw std::runtime_error("not implemented"); }
    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    type with_replaced_storage_type(const type &replacement_type) const;
  };

} // namespace dynd::ndt
} // namespace dynd
