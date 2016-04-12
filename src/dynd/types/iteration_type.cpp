//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/iteration_type.hpp>

using namespace std;
using namespace dynd;

ndt::iteration_type::iteration_type() : base_type(iteration_id, 0, 1, type_flag_symbolic, 0, 0, 0) {}

void ndt::iteration_type::print_type(ostream &o) const { o << "Iteration"; }

bool ndt::iteration_type::operator==(const base_type &rhs) const {
  return this == &rhs || rhs.get_id() == iteration_id;
}
