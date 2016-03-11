//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API array_type : public base_type {
  public:
    array_type();

    bool operator==(const base_type &rhs) const;

    virtual void data_construct(const char *arrmeta, char *data) const;

    virtual void data_destruct(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data)) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    static type make() { return type(new array_type(), false); }
  };

} // namespace dynd::ndt
} // namespace dynd
