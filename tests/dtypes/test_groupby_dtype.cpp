//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(GroupByDType, Create) {
    int data[] = {10,20,30};
    int by[] = {15,16,16};
    int groups[] = {15,16};
    ndobject g = groupby(data, by, make_categorical_dtype(groups));
    EXPECT_EQ(1, g.at(0).get_shape()[0]);
    EXPECT_EQ(2, g.at(1).get_shape()[0]);
    EXPECT_EQ(10, g.at(0,0).as<int>());
    EXPECT_EQ(20, g.at(1,0).as<int>());
    EXPECT_EQ(30, g.at(1,1).as<int>());
}
