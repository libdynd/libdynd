//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/new_adapt_type.hpp>
#include <dynd/types/builtin_type_properties.hpp>

using namespace std;
using namespace dynd;

ndt::new_adapt_type::new_adapt_type(const ndt::type &value_tp, const ndt::type &storage_tp, const nd::callable &forward,
                                    const nd::callable &inverse)
    : base_type(new_adapt_type_id, expr_kind, storage_tp.get_data_size(), storage_tp.get_data_alignment(),
                type_flag_none, storage_tp.get_arrmeta_size(), storage_tp.get_ndim(), storage_tp.get_strided_ndim()),
      m_value_tp(value_tp), m_storage_tp(storage_tp), m_forward(forward), m_inverse(inverse)
{
}

ndt::new_adapt_type::new_adapt_type(const nd::callable &forward, const nd::callable &inverse)
    : new_adapt_type(forward.get_type()->get_return_type(), forward.get_type()->get_pos_type(0), forward, inverse)
{
}

void ndt::new_adapt_type::print_type(ostream &o) const
{
  o << "adapt[";
  if (m_forward.is_null()) {
    o << "null";
  }
  else {
    o << m_forward;
  }
  o << ", ";
  if (m_inverse.is_null()) {
    o << "null";
  }
  else {
    o << m_inverse;
  }
  o << "]";
}

void ndt::new_adapt_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const
{
  m_storage_tp.print_data(o, arrmeta, data);
}

bool ndt::new_adapt_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }

  if (rhs.get_type_id() != new_adapt_type_id) {
    return false;
  }

  return false;
}

void ndt::new_adapt_type::get_dynamic_array_properties(std::map<std::string, nd::callable> &properties) const
{
  const type &udt = m_value_tp.get_dtype();
  if (!udt.is_builtin()) {
    udt.extended()->get_dynamic_array_properties(properties);
  }
  else {
    get_builtin_type_dynamic_array_properties(udt.get_type_id(), properties);
  }
}

void ndt::new_adapt_type::get_dynamic_array_functions(std::map<std::string, nd::callable> &functions) const
{
  const type &udt = m_value_tp.get_dtype();
  if (!udt.is_builtin()) {
    udt.extended()->get_dynamic_array_functions(functions);
  }
  else {
    // get_builtin_type_dynamic_array_functions(udt.get_type_id(),
    // out_functions, out_count);
  }
}
