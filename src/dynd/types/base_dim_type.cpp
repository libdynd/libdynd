//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

void ndt::base_dim_type::get_element_types(std::size_t ndim, const type **element_tp) const
{
  if (ndim > 0) {
    element_tp[0] = &m_element_tp;
    m_element_tp.extended<base_dim_type>()->get_element_types(ndim - 1, element_tp + 1);
  }
}

bool ndt::base_dim_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const
{
  if (get_id() != candidate_tp.get_id()) {
    return false;
  }

  return m_element_tp.match(candidate_tp.extended<base_dim_type>()->m_element_tp, tp_vars);
}
