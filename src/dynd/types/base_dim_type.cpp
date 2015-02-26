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

bool base_dim_type::match(const char *arrmeta, const ndt::type &candidate_tp,
                          const char *candidate_arrmeta,
                          std::map<nd::string, ndt::type> &tp_vars) const
{
  if (get_type_id() != candidate_tp.get_type_id()) {
    return false;
  }

  return m_element_tp.match(
      arrmeta, candidate_tp.extended<base_dim_type>()->m_element_tp,
      candidate_arrmeta, tp_vars);
}