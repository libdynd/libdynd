//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "inc_gtest.hpp"

#include <dynd/type.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

struct custom_type : ndt::base_type {
  custom_type(type_id_t tp_id, const nd::array &DYND_UNUSED(args))
      : base_type(tp_id, custom_kind, 0, 1, type_flag_none, 0, 0, 0)
  {
  }

  void print_type(ostream &o) const
  {
    o << "quaternion";
  }

  bool operator==(const ndt::base_type &rhs) const
  {
    return get_type_id() == rhs.get_type_id();
  }
};

TEST(Type, Registry)
{
  type_id_t tp_id = ndt::register_type<custom_type>("quaternion");

  const nd::array &a = nd::array();
  ndt::type tp = ndt::type::make(tp_id, a);
  EXPECT_EQ(tp_id, tp.get_type_id());
}
