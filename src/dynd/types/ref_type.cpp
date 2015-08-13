//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/ref_type.hpp>

using namespace std;
using namespace dynd;

ndt::ref_type::ref_type(const type &value_tp)
    : base_expr_type(ref_type_id, expr_kind, sizeof(nd::array),
                     sizeof(nd::array),
                     inherited_flags(value_tp.get_flags(), type_flag_zeroinit),
                     value_tp.get_arrmeta_size(), value_tp.get_ndim()),
      m_value_tp(value_tp)
{
}

ndt::ref_type::~ref_type()
{
}

bool ndt::ref_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }

  return rhs.get_type_id() == ref_type_id;
}

void ndt::ref_type::print_type(ostream &o) const
{
  o << "ref[" << m_value_tp << "]";
}

ndt::type ndt::ref_type::with_replaced_storage_type(const type &) const
{
  throw runtime_error("TODO: implement ref_type::with_replaced_storage_type");
}
