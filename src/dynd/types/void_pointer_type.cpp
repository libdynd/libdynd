//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/void_pointer_type.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

void ndt::void_pointer_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  uintptr_t target_ptr = *reinterpret_cast<const uintptr_t *>(data);
  o << "0x";
  hexadecimal_print(o, target_ptr);
}

void ndt::void_pointer_type::print_type(std::ostream &o) const { o << "pointer[void]"; }

bool ndt::void_pointer_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                                    const type &DYND_UNUSED(src_tp)) const
{
  return false;
}

bool ndt::void_pointer_type::operator==(const base_type &rhs) const
{
  return rhs.get_type_id() == void_pointer_type_id;
}
