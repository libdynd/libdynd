//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/any_kind_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API scalar_kind_type : public base_type {
  public:
    scalar_kind_type() : base_type(scalar_kind_id, 0, 0, type_flag_symbolic, 0, 0, 0) {}

    bool operator==(const base_type &rhs) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    void print_type(std::ostream &o) const;
  };

  template <>
  struct id_of<scalar_kind_type> : std::integral_constant<type_id_t, scalar_kind_id> {};

} // namespace dynd::ndt
} // namespace dynd
