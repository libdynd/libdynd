//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

ndt::base_dim_type::base_dim_type(type_id_t type_id, const type &element_tp, size_t data_size, size_t alignment,
                                  size_t element_arrmeta_offset, uint32_t flags, bool strided)
    : base_type(type_id, data_size, alignment, flags | type_flag_indexable,
                element_arrmeta_offset + element_tp.get_arrmeta_size(), 1 + element_tp.get_ndim(),
                strided ? (1 + element_tp.get_strided_ndim()) : 0),
      m_element_tp(element_tp), m_element_arrmeta_offset(element_arrmeta_offset)
{
}

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
