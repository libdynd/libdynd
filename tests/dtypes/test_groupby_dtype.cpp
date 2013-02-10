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
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(GroupByDType, Basic) {
    int data[] = {10,20,30};
    int by[] = {15,16,16};
    int groups[] = {15,16};
    ndobject g = groupby(data, by, make_categorical_dtype(groups));
    EXPECT_EQ(make_groupby_dtype(make_strided_array_dtype(make_dtype<int>()),
                        make_strided_array_dtype(make_convert_dtype(
                            make_categorical_dtype(groups), make_dtype<int>()))),
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
                            make_categorical_dtype(expected_groups), make_string_dtype()))),
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
                            make_categorical_dtype(arange(20)), make_dtype<int>()))),
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

TEST(GroupByDType, Struct) {
    const char *gender_cats_vals[] = {"F", "M"};
    ndobject gender_cats = ndobject(gender_cats_vals).cast_scalars(make_fixedstring_dtype(string_encoding_ascii, 1)).vals();

    // Create a simple structured array
    dtype d = make_fixedstruct_dtype(make_string_dtype(), "name", make_dtype<float>(), "height",
                    make_fixedstring_dtype(string_encoding_ascii, 1), "gender");
    ndobject a = make_strided_ndobject(5, d);
    const char *name_vals[] = {"Paul", "Jennifer", "Frank", "Louise", "Anne"};
    float height_vals[] = {171.5f, 156.25f, 177.0f, 164.75f, 170.5f};
    const char *gender_vals[] = {"M", "F", "M", "F", "F"};
    a.p("name").vals() = name_vals;
    a.p("height").vals() = height_vals;
    a.p("gender").vals() = gender_vals;

    // Group based on gender
    ndobject g = groupby(a, a.p("gender"));

    EXPECT_EQ(make_groupby_dtype(make_strided_array_dtype(d),
                        make_strided_array_dtype(make_convert_dtype(
                            make_categorical_dtype(gender_cats), gender_cats.get_udtype()))),
                    g.get_dtype());
    g = g.vals();
    EXPECT_EQ(2, g.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(3, g.at(0).get_shape()[0]);
    EXPECT_EQ(2, g.at(1).get_shape()[0]);
    EXPECT_EQ("Jennifer",   g.at(0,0).p("name").as<string>());
    EXPECT_EQ("Louise",     g.at(0,1).p("name").as<string>());
    EXPECT_EQ("Anne",       g.at(0,2).p("name").as<string>());
    EXPECT_EQ("Paul",       g.at(1,0).p("name").as<string>());
    EXPECT_EQ("Frank",      g.at(1,1).p("name").as<string>());
    EXPECT_EQ(156.25f,  g.at(0,0).p("height").as<float>());
    EXPECT_EQ(164.75f,  g.at(0,1).p("height").as<float>());
    EXPECT_EQ(170.5f,   g.at(0,2).p("height").as<float>());
    EXPECT_EQ(171.5f,   g.at(1,0).p("height").as<float>());
    EXPECT_EQ(177.0f,   g.at(1,1).p("height").as<float>());
}

TEST(GroupByDType, MismatchedSizes) {
    int data[] = {10,20,30};
    int by[] = {15,16,16,15};
    int groups[] = {15,16};
    EXPECT_THROW(groupby(data, by, make_categorical_dtype(groups)),
            runtime_error);
}

TEST(GroupByDType, StructUnsortedCats) {
    // The categories are not in alphabetical order
    const char *gender_cats_vals[] = {"M", "F"};
    ndobject gender_cats = ndobject(gender_cats_vals);

    // Create a simple structured array
    dtype d = make_fixedstruct_dtype(make_string_dtype(), "name", make_dtype<float>(), "height",
                    make_fixedstring_dtype(string_encoding_ascii, 1), "gender");
    ndobject a = make_strided_ndobject(5, d);
    const char *name_vals[] = {"Paul", "Jennifer", "Frank", "Louise", "Anne"};
    float height_vals[] = {171.5f, 156.25f, 177.0f, 164.75f, 170.5f};
    const char *gender_vals[] = {"M", "F", "M", "F", "F"};
    a.p("name").vals() = name_vals;
    a.p("height").vals() = height_vals;
    a.p("gender").vals() = gender_vals;

    // Group based on gender
    ndobject g = groupby(a, a.p("gender"), make_categorical_dtype(gender_cats));

    EXPECT_EQ(make_groupby_dtype(make_strided_array_dtype(d),
                        make_strided_array_dtype(make_convert_dtype(
                            make_categorical_dtype(gender_cats), a.p("gender").get_udtype()))),
                    g.get_dtype());
    g = g.vals();
    EXPECT_EQ(2, g.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(2, g.at(0).get_shape()[0]);
    EXPECT_EQ(3, g.at(1).get_shape()[0]);
    EXPECT_EQ("Paul",       g.at(0,0).p("name").as<string>());
    EXPECT_EQ("Frank",      g.at(0,1).p("name").as<string>());
    EXPECT_EQ("Jennifer",   g.at(1,0).p("name").as<string>());
    EXPECT_EQ("Louise",     g.at(1,1).p("name").as<string>());
    EXPECT_EQ("Anne",       g.at(1,2).p("name").as<string>());
    EXPECT_EQ(171.5f,   g.at(0,0).p("height").as<float>());
    EXPECT_EQ(177.0f,   g.at(0,1).p("height").as<float>());
    EXPECT_EQ(156.25f,  g.at(1,0).p("height").as<float>());
    EXPECT_EQ(164.75f,  g.at(1,1).p("height").as<float>());
    EXPECT_EQ(170.5f,   g.at(1,2).p("height").as<float>());
}

