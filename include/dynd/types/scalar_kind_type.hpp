//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

namespace dynd {
namespace ndt {

  class DYND_API scalar_kind_type : public base_type {
  public:
    scalar_kind_type();

    virtual ~scalar_kind_type();

    bool operator==(const base_type &rhs) const;

    bool match(const char *arrmeta, const type &candidate_tp,
               const char *candidate_arrmeta,
               std::map<std::string, type> &tp_vars) const;

    void print_type(std::ostream &o) const;

    static type make() { return type(new scalar_kind_type(), false); }
  };

} // namespace dynd::ndt
} // namespace dynd
