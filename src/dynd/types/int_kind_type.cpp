//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/int_kind_type.hpp>

using namespace std;
using namespace dynd;

bool ndt::int_kind_type::match(const type &candidate_tp, std::map<std::string, type> &DYND_UNUSED(tp_vars)) const {
  return candidate_tp.get_base_id() == int_kind_id;
}

void ndt::int_kind_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                    const char *DYND_UNUSED(data)) const {
  throw std::runtime_error("cannot print data of array_type");
}

void ndt::int_kind_type::print_type(ostream &o) const { o << "Int"; }

bool ndt::int_kind_type::operator==(const base_type &rhs) const { return this == &rhs || rhs.get_id() == int_kind_id; }
