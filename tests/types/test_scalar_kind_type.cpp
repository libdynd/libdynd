//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/types/scalar_kind_type.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

TEST(ScalarKindType, Basic) {
  ndt::type scalar_kind_tp = ndt::make_type<ndt::scalar_kind_type>();
  EXPECT_EQ(scalar_kind_id, scalar_kind_tp.get_id());
  EXPECT_TRUE(scalar_kind_tp.is_scalar());
  EXPECT_TRUE(scalar_kind_tp.is_symbolic());
  EXPECT_EQ(0, scalar_kind_tp.get_ndim());
  EXPECT_EQ("Scalar", scalar_kind_tp.str());
  EXPECT_EQ(scalar_kind_tp, ndt::type(scalar_kind_tp.str()));
}

TEST(ScalarKindType, Match) {
  ndt::type tp = ndt::make_type<ndt::scalar_kind_type>();
  EXPECT_TRUE(tp.match(ndt::type("Scalar")));
  EXPECT_TRUE(tp.match(ndt::type("int32")));
  EXPECT_TRUE(tp.match(ndt::type("float64")));
  EXPECT_TRUE(tp.match(ndt::type("(int32, float64)")));
  EXPECT_TRUE(tp.match(ndt::type("(int32, float64, ...)")));
  EXPECT_FALSE(tp.match(ndt::type("Any")));
  EXPECT_FALSE(tp.match(ndt::type("4 * int32")));
  EXPECT_FALSE(tp.match(ndt::type("Fixed * Any")));
  EXPECT_FALSE(tp.match(ndt::type("Dims... * float64")));
}

TEST(ScalarKindType, IDOf) { EXPECT_EQ(scalar_kind_id, ndt::id_of<ndt::scalar_kind_type>::value); }
