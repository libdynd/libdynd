//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/state_type.hpp>

using namespace std;
using namespace dynd;

ndt::state_type::state_type() : base_type(state_id, 0, 1, type_flag_symbolic, 0, 0, 0) {}

void ndt::state_type::print_type(ostream &o) const { o << "State"; }

bool ndt::state_type::operator==(const base_type &rhs) const { return this == &rhs || rhs.get_id() == state_id; }
