//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/array_range.hpp>
#include <dynd/types/groupby_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/struct_type.hpp>

#include "dynd_assertions.hpp"

using namespace std;
using namespace dynd;

TEST(GroupByDType, Basic)
{
  int data[] = {10, 20, 30};
  int by[] = {15, 16, 16};
  int groups[] = {15, 16};
  nd::array g = nd::groupby(data, by, ndt::categorical_type::make(groups));
  EXPECT_EQ(ndt::make_groupby(
                ndt::make_fixed_dim(3, ndt::type::make<int>()),
                ndt::make_fixed_dim(3, ndt::convert_type::make(
                                           ndt::categorical_type::make(groups),
                                           ndt::type::make<int>()))),
            g.get_type());
  g = g.eval();
  EXPECT_EQ(1, g(0, irange()).get_shape()[0]);
  EXPECT_EQ(2, g(1, irange()).get_shape()[0]);
  EXPECT_EQ(10, g(0, 0).as<int>());
  EXPECT_EQ(20, g(1, 0).as<int>());
  EXPECT_EQ(30, g(1, 1).as<int>());
}

TEST(GroupByDType, BasicDeduceGroups)
{
  const char *data[] = {"a", "test", "is", "here", "now"};
  const char *by[] = {"beta", "alpha", "beta", "beta", "alpha"};
  nd::array g = nd::groupby(data, by);
  const char *expected_groups[] = {"alpha", "beta"};
  EXPECT_EQ(
      ndt::make_groupby(ndt::make_fixed_dim(5, ndt::string_type::make()),
                        ndt::make_fixed_dim(
                            5, ndt::convert_type::make(
                                   ndt::categorical_type::make(expected_groups),
                                   ndt::string_type::make()))),
      g.get_type());
  g = g.eval();
  EXPECT_EQ(2, g(0, irange()).get_shape()[0]);
  EXPECT_EQ(3, g(1, irange()).get_shape()[0]);
  EXPECT_EQ("test", g(0, 0).as<std::string>());
  EXPECT_EQ("now", g(0, 1).as<std::string>());
  EXPECT_EQ("a", g(1, 0).as<std::string>());
  EXPECT_EQ("is", g(1, 1).as<std::string>());
  EXPECT_EQ("here", g(1, 2).as<std::string>());
}

TEST(GroupByDType, MediumDeduceGroups)
{
  nd::array data = nd::range(100);
  nd::array by = nd::empty<int[100]>();
  // Since at this point dynd doesn't have a very sophisticated
  // calculation mechanism, construct by as a series of runs
  by(0 <= irange() < 10).vals() = nd::range(10);
  by(10 <= irange() < 25).vals() = nd::range(15);
  by(25 <= irange() < 35).vals() = nd::range(10);
  by(35 <= irange() < 55).vals() = nd::range(20);
  by(55 <= irange() < 60).vals() = nd::range(5);
  by(60 <= irange() < 80).vals() = nd::range(20);
  by(80 <= irange() < 95).vals() = nd::range(15);
  by(95 <= irange() < 100).vals() = nd::range(5);
  nd::array g = nd::groupby(data, by);
  EXPECT_EQ(
      ndt::make_groupby(ndt::make_fixed_dim(100, ndt::type::make<int>()),
                        ndt::make_fixed_dim(
                            100, ndt::convert_type::make(
                                     ndt::categorical_type::make(nd::range(20)),
                                     ndt::type::make<int>()))),
      g.get_type());
  EXPECT_EQ(g.get_shape()[0], 20);
  int group_0[] = {0, 10, 25, 35, 55, 60, 80, 95};
  int group_6[] = {6, 16, 31, 41, 66, 86};
  int group_9[] = {9, 19, 34, 44, 69, 89};
  int group_10[] = {20, 45, 70, 90};
  int group_15[] = {50, 75};
  int group_19[] = {54, 79};
  g = g.eval();
  EXPECT_ARRAY_VALS_EQ(nd::array(group_0), g(0, irange()).eval());
  EXPECT_ARRAY_VALS_EQ(nd::array(group_6), g(6, irange()).eval());
  EXPECT_ARRAY_VALS_EQ(nd::array(group_9), g(9, irange()).eval());
  EXPECT_ARRAY_VALS_EQ(nd::array(group_10), g(10, irange()).eval());
  EXPECT_ARRAY_VALS_EQ(nd::array(group_15), g(15, irange()).eval());
  EXPECT_ARRAY_VALS_EQ(nd::array(group_19), g(19, irange()).eval());
}

TEST(GroupByDType, Struct)
{
  const char *gender_cats_vals[] = {"F", "M"};
  nd::array gender_cats =
      nd::array(gender_cats_vals)
          .ucast(ndt::fixed_string_type::make(1, string_encoding_ascii))
          .eval();

  // Create a simple structured array
  ndt::type d = ndt::struct_type::make(
      {"name", "height", "gender"},
      {ndt::string_type::make(), ndt::type::make<float>(),
       ndt::fixed_string_type::make(1, string_encoding_ascii)});
  nd::array a = nd::empty(5, d);
  const char *name_vals[] = {"Paul", "Jennifer", "Frank", "Louise", "Anne"};
  float height_vals[] = {171.5f, 156.25f, 177.0f, 164.75f, 170.5f};
  const char *gender_vals[] = {"M", "F", "M", "F", "F"};
  a.p("name").vals() = name_vals;
  a.p("height").vals() = height_vals;
  a.p("gender").vals() = gender_vals;

  // Group based on gender
  nd::array g = nd::groupby(a, a.p("gender"));

  EXPECT_EQ(
      ndt::make_groupby(
          ndt::make_fixed_dim(5, d),
          ndt::make_fixed_dim(5, ndt::convert_type::make(
                                     ndt::categorical_type::make(gender_cats),
                                     gender_cats.get_dtype()))),
      g.get_type());
  g = g.eval();
  EXPECT_EQ(2, g.at_array(0, NULL).get_shape()[0]);
  EXPECT_EQ(3, g(0, irange()).get_shape()[0]);
  EXPECT_EQ(2, g(1, irange()).get_shape()[0]);
  EXPECT_EQ("Jennifer", g(0, 0).p("name").as<std::string>());
  EXPECT_EQ("Louise", g(0, 1).p("name").as<std::string>());
  EXPECT_EQ("Anne", g(0, 2).p("name").as<std::string>());
  EXPECT_EQ("Paul", g(1, 0).p("name").as<std::string>());
  EXPECT_EQ("Frank", g(1, 1).p("name").as<std::string>());
  EXPECT_EQ(156.25f, g(0, 0).p("height").as<float>());
  EXPECT_EQ(164.75f, g(0, 1).p("height").as<float>());
  EXPECT_EQ(170.5f, g(0, 2).p("height").as<float>());
  EXPECT_EQ(171.5f, g(1, 0).p("height").as<float>());
  EXPECT_EQ(177.0f, g(1, 1).p("height").as<float>());
}

TEST(GroupByDType, StructSubset)
{
  // Create a simple structured array
  ndt::type d = ndt::struct_type::make(
      {"lastname", "firstname", "gender"},
      {ndt::string_type::make(), ndt::string_type::make(),
       ndt::fixed_string_type::make(1, string_encoding_ascii)});
  nd::array a = nd::empty(7, d);
  const char *lastname_vals[] = {"Wiebe",   "Friesen", "Klippenstein", "Wiebe",
                                 "Friesen", "Friesen", "Friesen"};
  const char *firstname_vals[] = {"Paul", "Jennifer", "Frank", "Louise",
                                  "Jake", "Arthur",   "Anne"};
  const char *gender_vals[] = {"M", "F", "M", "F", "M", "M", "F"};
  a.p("lastname").vals() = lastname_vals;
  a.p("firstname").vals() = firstname_vals;
  a.p("gender").vals() = gender_vals;

  // Group based on gender
  nd::array g = nd::groupby(a, a.p("gender"));

  g = g.eval();
  EXPECT_EQ(2, g.at_array(0, NULL).get_shape()[0]);
  EXPECT_EQ(3, g(0, irange()).get_shape()[0]);
  EXPECT_EQ(4, g(1, irange()).get_shape()[0]);
  EXPECT_EQ("Jennifer", g(0, 0).p("firstname").as<std::string>());
  EXPECT_EQ("Louise", g(0, 1).p("firstname").as<std::string>());
  EXPECT_EQ("Anne", g(0, 2).p("firstname").as<std::string>());
  EXPECT_EQ("Paul", g(1, 0).p("firstname").as<std::string>());
  EXPECT_EQ("Frank", g(1, 1).p("firstname").as<std::string>());
  EXPECT_EQ("Jake", g(1, 2).p("firstname").as<std::string>());
  EXPECT_EQ("Arthur", g(1, 3).p("firstname").as<std::string>());

  // Group based on last name, gender
  g = nd::groupby(
      a, a(irange(), irange(0, 3, 2))); // a(irange(), {"lastname", "gender"})

  // Validate the list of groups it produced

  /*
      Todo: Fix this test
  nd::array groups_list = g.p("groups");
  EXPECT_EQ(ndt::make_fixed_dim(
                5, ndt::struct_type::make({"lastname", "gender"},
                                          {ndt::string_type::make(),
                                           ndt::fixed_string_type::make(
                                               1, string_encoding_ascii)})),
            groups_list.get_type());
  EXPECT_EQ(5, groups_list.get_shape()[0]);
  EXPECT_EQ("Friesen", groups_list(0, 0).as<std::string>());
  EXPECT_EQ("F", groups_list(0, 1).as<std::string>());
  EXPECT_EQ("Friesen", groups_list(1, 0).as<std::string>());
  EXPECT_EQ("M", groups_list(1, 1).as<std::string>());
  EXPECT_EQ("Klippenstein", groups_list(2, 0).as<std::string>());
  EXPECT_EQ("M", groups_list(2, 1).as<std::string>());
  EXPECT_EQ("Wiebe", groups_list(3, 0).as<std::string>());
  EXPECT_EQ("F", groups_list(3, 1).as<std::string>());
  EXPECT_EQ("Wiebe", groups_list(4, 0).as<std::string>());
  EXPECT_EQ("M", groups_list(4, 1).as<std::string>());
*/

  /*
      Todo: Fix this test

      g = g.eval();
      EXPECT_EQ(5, g.at_array(0, NULL).get_shape()[0]);
      EXPECT_EQ(2, g(0, irange()).get_shape()[0]);
      EXPECT_EQ(2, g(1, irange()).get_shape()[0]);
      EXPECT_EQ(1, g(2, irange()).get_shape()[0]);
      EXPECT_EQ(1, g(3, irange()).get_shape()[0]);
      EXPECT_EQ(1, g(4, irange()).get_shape()[0]);
      EXPECT_EQ("Jennifer",   g(0,0).p("firstname").as<std::string>());
      EXPECT_EQ("Anne",       g(0,1).p("firstname").as<std::string>());
      EXPECT_EQ("Jake",       g(1,0).p("firstname").as<std::string>());
      EXPECT_EQ("Arthur",     g(1,1).p("firstname").as<std::string>());
      EXPECT_EQ("Frank",      g(2,0).p("firstname").as<std::string>());
      EXPECT_EQ("Louise",     g(3,0).p("firstname").as<std::string>());
      EXPECT_EQ("Paul",       g(4,0).p("firstname").as<std::string>());
  */
}

TEST(GroupByDType, MismatchedSizes)
{
  int data[] = {10, 20, 30};
  int by[] = {15, 16, 16, 15};
  int groups[] = {15, 16};
  EXPECT_THROW(nd::groupby(data, by, ndt::categorical_type::make(groups)),
               runtime_error);
}

TEST(GroupByDType, StructUnsortedCats)
{
  // The categories are not in alphabetical order
  const char *gender_cats_vals[] = {"M", "F"};
  nd::array gender_cats = nd::array(gender_cats_vals);

  // Create a simple structured array
  ndt::type d = ndt::struct_type::make(
      {"name", "height", "gender"},
      {ndt::string_type::make(), ndt::type::make<float>(),
       ndt::fixed_string_type::make(1, string_encoding_ascii)});
  nd::array a = nd::empty(5, d);
  const char *name_vals[] = {"Paul", "Jennifer", "Frank", "Louise", "Anne"};
  float height_vals[] = {171.5f, 156.25f, 177.0f, 164.75f, 170.5f};
  const char *gender_vals[] = {"M", "F", "M", "F", "F"};
  a.p("name").vals() = name_vals;
  a.p("height").vals() = height_vals;
  a.p("gender").vals() = gender_vals;

  // Group based on gender
  nd::array g =
      nd::groupby(a, a.p("gender"), ndt::categorical_type::make(gender_cats));

  EXPECT_EQ(
      ndt::make_groupby(
          ndt::make_fixed_dim(5, d),
          ndt::make_fixed_dim(5, ndt::convert_type::make(
                                     ndt::categorical_type::make(gender_cats),
                                     a.p("gender").get_dtype()))),
      g.get_type());
  g = g.eval();
  EXPECT_EQ(2, g.at_array(0, NULL).get_shape()[0]);
  EXPECT_EQ(2, g(0, irange()).get_shape()[0]);
  EXPECT_EQ(3, g(1, irange()).get_shape()[0]);
  EXPECT_EQ("Paul", g(0, 0).p("name").as<std::string>());
  EXPECT_EQ("Frank", g(0, 1).p("name").as<std::string>());
  EXPECT_EQ("Jennifer", g(1, 0).p("name").as<std::string>());
  EXPECT_EQ("Louise", g(1, 1).p("name").as<std::string>());
  EXPECT_EQ("Anne", g(1, 2).p("name").as<std::string>());
  EXPECT_EQ(171.5f, g(0, 0).p("height").as<float>());
  EXPECT_EQ(177.0f, g(0, 1).p("height").as<float>());
  EXPECT_EQ(156.25f, g(1, 0).p("height").as<float>());
  EXPECT_EQ(164.75f, g(1, 1).p("height").as<float>());
  EXPECT_EQ(170.5f, g(1, 2).p("height").as<float>());
}
