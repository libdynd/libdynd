//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/base_dim_type.hpp>

#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

base_dim_type::~base_dim_type() {
}

bool base_dim_type::matches(const char *arrmeta, const ndt::type &other,
                            std::map<nd::string, ndt::type> &tp_vars) const
{
  if (other.is_symbolic()) {
    return other.matches(arrmeta, ndt::type(this, true), tp_vars);
  }

  return m_element_tp.matches(arrmeta, other.extended<base_dim_type>()->m_element_tp,
    tp_vars);
}