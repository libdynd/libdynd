//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>

namespace dynd {

struct iteration_t {
  size_t ndim;
  size_t *index;
};

namespace ndt {

  class DYNDT_API iteration_type : public base_type {
  public:
    iteration_type();

    void print_type(std::ostream &o) const;

    bool operator==(const base_type &rhs) const;
  };

  template <>
  struct traits<iteration_t> {
    static type equivalent() { return make_type<iteration_type>(); }
  };

} // namespace dynd::ndt
} // namespace dynd
