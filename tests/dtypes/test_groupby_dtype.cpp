//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/ndobject_arange.hpp>
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

TEST(GroupByDType, BasicDeduceGroups) {
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

TEST(GroupByDType, MediumDeduceGroups) {
    ndobject data = arange(100);
    ndobject by = make_strided_ndobject(100, make_dtype<int>());
    // Since at this point dynd doesn't have a very sophisticated
    // calculation mechanism, construct by as a series of runs
    by.at(0 <= irange() < 10).vals() = arange(10);
    by.at(10 <= irange() < 25).vals() = arange(15);
    by.at(25 <= irange() < 35).vals() = arange(10);
    by.at(35 <= irange() < 55).vals() = arange(20);
    by.at(55 <= irange() < 60).vals() = arange(5);
    by.at(60 <= irange() < 80).vals() = arange(20);
    by.at(80 <= irange() < 95).vals() = arange(15);
    by.at(95 <= irange() < 100).vals() = arange(5);
    ndobject g = groupby(data, by);
    EXPECT_EQ(make_groupby_dtype(make_strided_array_dtype(make_dtype<int>()),
                        make_strided_array_dtype(make_convert_dtype(
                            make_categorical_dtype(arange(20)), make_dtype<int>())),
                        make_categorical_dtype(arange(20))),
                    g.get_dtype());
    EXPECT_EQ(g.get_shape()[0], 20);
    int group_0[] =  { 0, 10, 25, 35, 55, 60, 80, 95};
    int group_6[] =  { 6, 16, 31, 41,     66, 86    };
    int group_9[] =  { 9, 19, 34, 44,     69, 89    };
    int group_10[] = {    20,     45,     70, 90    };
    int group_15[] = {            50,     75        };
    int group_19[] = {            54,     79        };
    g = g.vals();
    EXPECT_TRUE(ndobject(group_0).equals_exact(g.at(0).vals()));
    EXPECT_TRUE(ndobject(group_6).equals_exact(g.at(6).vals()));
    EXPECT_TRUE(ndobject(group_9).equals_exact(g.at(9).vals()));
    EXPECT_TRUE(ndobject(group_10).equals_exact(g.at(10).vals()));
    EXPECT_TRUE(ndobject(group_15).equals_exact(g.at(15).vals()));
    EXPECT_TRUE(ndobject(group_19).equals_exact(g.at(19).vals()));
}