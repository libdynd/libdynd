//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/adapt_type.hpp>

using namespace std;
using namespace dynd;

void ndt::adapt_type::print_type(ostream &o) const {
  o << "adapt[";
  if (m_forward.is_null()) {
    o << "null";
  } else {
    o << m_forward;
  }
  o << ", ";
  if (m_inverse.is_null()) {
    o << "null";
  } else {
    o << m_inverse;
  }
  o << "]";
}

void ndt::adapt_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const {
  m_storage_tp.print_data(o, arrmeta, data);
}

bool ndt::adapt_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  }

  if (rhs.get_id() != adapt_id) {
    return false;
  }

  return false;
}
