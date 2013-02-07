//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/groupby_dtype.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(GroupByDType, Basic) {
    int data[] = {10,20,30};
    int by[] = {15,16,16};
    int groups[] = {15,16};
    ndobject g = groupby(data, by, make_categorical_dtype(groups));
    EXPECT_EQ(make_groupby_dtype(make_strided_array_dtype(make_dtype<int>()),
                        make_strided_array_dtype(make_convert_dtype(
                            make_categorical_dtype(groups), make_dtype<int>())),
                        make_categorical_dtype(groups)),
                    g.get_dtype());
    g = g.vals();
    EXPECT_EQ(1, g.at(0).get_shape()[0]);
    EXPECT_EQ(2, g.at(1).get_shape()[0]);
    EXPECT_EQ(10, g.at(0,0).as<int>());
    EXPECT_EQ(20, g.at(1,0).as<int>());
    EXPECT_EQ(30, g.at(1,1).as<int>());
}

TEST(GroupByDType, DeduceGroups) {
    const char *data[] = {"a", "test", "is", "here", "now"};
    const char *by[] = {"beta", "alpha", "beta", "beta", "alpha"};
    ndobject g = groupby(data, by);
    const char *expected_groups[] = {"alpha", "beta"};
    EXPECT_EQ(make_groupby_dtype(make_strided_array_dtype(make_string_dtype()),
                        make_strided_array_dtype(make_convert_dtype(
                            make_categorical_dtype(expected_groups), make_string_dtype())),
                        make_categorical_dtype(expected_groups)),
                    g.get_dtype());
    g = g.vals();
    EXPECT_EQ(2, g.at(0).get_shape()[0]);
    EXPECT_EQ(3, g.at(1).get_shape()[0]);
    EXPECT_EQ("test",   g.at(0,0).as<string>());
    EXPECT_EQ("now",    g.at(0,1).as<string>());
    EXPECT_EQ("a",      g.at(1,0).as<string>());
    EXPECT_EQ("is",     g.at(1,1).as<string>());
    EXPECT_EQ("here",   g.at(1,2).as<string>());
}
