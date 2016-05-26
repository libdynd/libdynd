//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API array_type : public base_type {
  public:
    array_type(type_id_t new_id)
        : base_type(new_id, array_id, sizeof(void *), alignof(void *), type_flag_construct | type_flag_destructor, 0, 0,
                    0) {}

    bool operator==(const base_type &rhs) const;

    virtual void data_construct(const char *arrmeta, char *data) const;

    virtual void data_destruct(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data)) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;
  };

  template <>
  struct id_of<array_type> : std::integral_constant<type_id_t, array_id> {};

} // namespace dynd::ndt
} // namespace dynd
