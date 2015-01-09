//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/static_type_instances.hpp>

namespace dynd {

/**
 * A type whose that represents a symbolic type.
 */
class sym_type_type : public base_type {
  ndt::type m_sym_tp;

public:
  sym_type_type(const ndt::type &sym_tp);

  virtual ~sym_type_type();

  const ndt::type &get_sym_type() const {
    return m_sym_tp;
  }

  void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

  void print_type(std::ostream &o) const;

  bool operator==(const base_type &rhs) const;
};

namespace ndt {
  /** Returns type "type" */
  inline const ndt::type make_sym_type_type(const ndt::type &sym_tp)
  {
    return ndt::type(new sym_type_type(sym_tp), false);
  }
} // namespace ndt

} // namespace dynd
