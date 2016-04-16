//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>

namespace dynd {
namespace nd {

  struct state {
    size_t ndim;
    size_t *index;
  };

} // namespace dynd::nd

namespace ndt {

  class DYNDT_API state_type : public base_type {
  public:
    state_type();

    void print_type(std::ostream &o) const;

    bool operator==(const base_type &rhs) const;
  };

  template <>
  struct traits<nd::state> {
    static type equivalent() { return make_type<state_type>(); }
  };

} // namespace dynd::ndt
} // namespace dynd
