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
    scalar_kind_type();

    bool operator==(const base_type &rhs) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    void print_type(std::ostream &o) const;
  };

} // namespace dynd::ndt
} // namespace dynd
