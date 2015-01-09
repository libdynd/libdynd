//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/sym_type_type.hpp>

using namespace std;
using namespace dynd;

sym_type_type::sym_type_type(const ndt::type &sym_tp)
    : base_type(sym_type_type_id, symbolic_kind, 0, 1, type_flag_symbolic, 0, 0,
                0),
      m_sym_tp(sym_tp)
{
}

sym_type_type::~sym_type_type() {}

void sym_type_type::print_data(std::ostream &DYND_UNUSED(o),
                               const char *DYND_UNUSED(arrmeta),
                               const char *DYND_UNUSED(data)) const
{
  throw type_error("Cannot store data of typevar type");
}

void sym_type_type::print_type(std::ostream &o) const
{
  o << "Type[" << m_sym_tp << "]";
}

bool sym_type_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  } else if (rhs.get_type_id() != sym_type_type_id) {
    return false;
  } else {
    return m_sym_tp == static_cast<const sym_type_type *>(&rhs)->m_sym_tp;
  }
}
