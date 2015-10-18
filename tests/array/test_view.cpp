//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"
#include "../test_memory.hpp"

#include <dynd/array_range.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/view.hpp>

using namespace std;
using namespace dynd;

TEST(View, SameType)
{
  // When the type is the same, nd::view should return the
  // exact same array
  nd::array a = nd::empty("5 * 3 * int32");
  nd::array b;

  b = nd::view(a, a.get_type());
  EXPECT_EQ(a.get_ndo(), b.get_ndo());

  // Slicing with : keeps the same fixed dim type
  a = a(irange(), irange());
  EXPECT_EQ(ndt::type("5 * 3 * int32"), a.get_type());
  b = nd::view(a, a.get_type());
  EXPECT_EQ(a.get_ndo(), b.get_ndo());

  // test also with a var dim and string type
  a = parse_json("3 * var * string", "[[\"this\", \"is\", \"for\"], [\"testing\"], []]");
  b = nd::view(a, a.get_type());
  EXPECT_EQ(a.get_ndo(), b.get_ndo());
}

TEST(View, Errors)
{
  nd::array a = nd::empty("5 * 3 * int32");

  // Shape mismatches
  EXPECT_THROW(nd::view(a, ndt::type("Fixed * 2 * int32")), type_error);
  EXPECT_THROW(nd::view(a, ndt::type("5 * 2 * int32")), type_error);
  EXPECT_THROW(nd::view(a, ndt::type("6 * 3 * int32")), type_error);
  // DType mismatches
  EXPECT_THROW(nd::view(a, ndt::type("5 * 3 * uint64")), type_error);
}

TEST(View, AsBytes)
{
  nd::array a, b;
  const bytes_type_arrmeta *btd_meta;
  const bytes_type_data *btd;

  // View a scalar as bytes
  a = (int32_t)100;
  b = nd::view(a, ndt::bytes_type::make(4));
  ASSERT_EQ(b.get_type(), ndt::bytes_type::make(4));
  // Confirm the bytes arrmeta points to the right data reference
  btd_meta = reinterpret_cast<const bytes_type_arrmeta *>(b.get_arrmeta());
  EXPECT_EQ(btd_meta->blockref, a.get_data_memblock().get());
  // Confirm it's pointing to the right memory with the right size
  btd = reinterpret_cast<const bytes_type_data *>(b.get_readonly_originptr());
  EXPECT_EQ(a.get_readonly_originptr(), btd->begin());
  EXPECT_EQ(4, btd->end() - btd->begin());

  // View a 1D array as bytes
  double a_data[2] = {1, 2};
  a = a_data;
  b = nd::view(a, ndt::bytes_type::make(1));
  ASSERT_EQ(b.get_type(), ndt::bytes_type::make(1));
  // Confirm the bytes arrmeta points to the right data reference
  btd_meta = reinterpret_cast<const bytes_type_arrmeta *>(b.get_arrmeta());
  EXPECT_EQ(btd_meta->blockref, a.get_data_memblock().get());
  // Confirm it's pointing to the right memory with the right size
  btd = reinterpret_cast<const bytes_type_data *>(b.get_readonly_originptr());
  EXPECT_EQ(a.get_readonly_originptr(), btd->begin());
  EXPECT_EQ(2 * 8, btd->end() - btd->begin());

  // View a 2D array as bytes
  double a_data2[2][3] = {{1, 2, 3}, {1, 2, 5}};
  a = a_data2;
  b = nd::view(a, ndt::bytes_type::make(2));
  ASSERT_EQ(b.get_type(), ndt::bytes_type::make(2));
  // Confirm the bytes arrmeta points to the right data reference
  btd_meta = reinterpret_cast<const bytes_type_arrmeta *>(b.get_arrmeta());
  EXPECT_EQ(btd_meta->blockref, a.get_data_memblock().get());
  // Confirm it's pointing to the right memory with the right size
  btd = reinterpret_cast<const bytes_type_data *>(b.get_readonly_originptr());
  EXPECT_EQ(a.get_readonly_originptr(), btd->begin());
  EXPECT_EQ(2 * 3 * 8, btd->end() - btd->begin());
  EXPECT_THROW(nd::view(a(irange(), irange(0, 2)), ndt::bytes_type::make(1)), type_error);

  // View an array with var outer dimension
  a = parse_json("var * 2 * int16", "[[1, 2], [3, 4], [5, 6]]");
  b = nd::view(a, ndt::bytes_type::make(1));
  ASSERT_EQ(b.get_type(), ndt::bytes_type::make(1));
  // Confirm the bytes arrmeta points to the right data reference
  btd_meta = reinterpret_cast<const bytes_type_arrmeta *>(b.get_arrmeta());
  const var_dim_type_arrmeta *vdt_meta = reinterpret_cast<const var_dim_type_arrmeta *>(a.get_arrmeta());
  EXPECT_EQ(btd_meta->blockref, vdt_meta->blockref);
  // Confirm it's pointing to the right memory with the right size
  const var_dim_type_data *vdt_data = reinterpret_cast<const var_dim_type_data *>(a.get_readonly_originptr());
  btd = reinterpret_cast<const bytes_type_data *>(b.get_readonly_originptr());
  EXPECT_EQ(vdt_data->begin, btd->begin());
  EXPECT_EQ(2 * 3 * 2, btd->end() - btd->begin());
}

/*
// TODO: With cstruct gone, this test as is doesn't make sense, need to work out
//       what is reasonable
TEST(View, StructAsBytes) {
    nd::array a, b;

    a = parse_json(
        "var * {ix : int64, dt : date, rl : int64, v : float64, vt : "
        "string[4], st : float64, tr : float64, r : float64}",
        "[[150, \"2002-06-27\", 391, 10.9307, \"abc\", 10, 2.2799, -10.6904]]");
    a = a(irange());
    // View this 1D array of struct as bytes
    b = nd::view(a, ndt::make_bytes(1));
    EXPECT_EQ(ndt::make_bytes(1), b.get_type());
    // View it back as a struct
    b = nd::view(
        b, ndt::type("{ix : int64, dt : date, rl : int64, v : float64, vt : "
                     "string[4], st : float64, tr : float64, r : float64}"));
    EXPECT_TRUE(b.equals_exact(a(0)));
}
*/

TEST(View, FromBytes)
{
  nd::array a, b;
  const bytes_type_arrmeta *btd_meta;
  const bytes_type_data *btd;

  double x = 3.25;
  a = nd::make_bytes_array(reinterpret_cast<const char *>(&x), sizeof(x), 8);
  ASSERT_EQ(ndt::bytes_type::make(8), a.get_type());
  b = nd::view(a, ndt::type::make<double>());
  EXPECT_EQ(3.25, b.as<double>());
  btd_meta = reinterpret_cast<const bytes_type_arrmeta *>(a.get_arrmeta());
  btd = reinterpret_cast<const bytes_type_data *>(a.get_readonly_originptr());
  if (btd_meta->blockref != NULL) {
    EXPECT_EQ(btd_meta->blockref, b.get_ndo()->data.ref);
  } else {
    EXPECT_EQ(a.get_data_memblock().get(), b.get_ndo()->data.ref);
  }
  EXPECT_EQ(btd->begin(), b.get_readonly_originptr());

  float y[3] = {1.f, 2.5f, -1.25f};
  a = nd::make_bytes_array(reinterpret_cast<const char *>(&y), sizeof(y), 4);
  ASSERT_EQ(ndt::bytes_type::make(4), a.get_type());
  b = nd::view(a, ndt::type("Fixed * float32"));
  EXPECT_EQ(1.f, b(0).as<float>());
  EXPECT_EQ(2.5f, b(1).as<float>());
  EXPECT_EQ(-1.25f, b(2).as<float>());
  btd_meta = reinterpret_cast<const bytes_type_arrmeta *>(a.get_arrmeta());
  btd = reinterpret_cast<const bytes_type_data *>(a.get_readonly_originptr());
  if (btd_meta->blockref != NULL) {
    EXPECT_EQ(btd_meta->blockref, b.get_ndo()->data.ref);
  } else {
    EXPECT_EQ(a.get_data_memblock().get(), b.get_ndo()->data.ref);
  }
  EXPECT_EQ(btd->begin(), b.get_readonly_originptr());
}

TEST(View, WeakerAlignment)
{
  nd::array a, b;

  int64_t aval = 0x0102030405060708LL;
  a = nd::make_bytes_array(reinterpret_cast<const char *>(&aval), sizeof(aval), sizeof(aval));
  b = nd::view(a, ndt::type("2 * int32"));
#ifdef DYND_BIG_ENDIAN
  EXPECT_EQ(0x01020304, b(0).as<int32_t>());
  EXPECT_EQ(0x05060708, b(1).as<int32_t>());
#else
  EXPECT_EQ(0x05060708, b(0).as<int32_t>());
  EXPECT_EQ(0x01020304, b(1).as<int32_t>());
#endif
}

TEST(View, StringAsBytes)
{
  nd::array a, b;

  a = parse_json("string", "\"\\U00024B62\"");
  b = nd::view(a, "bytes");
  const bytes_type_data *btd = reinterpret_cast<const bytes_type_data *>(b.get_readonly_originptr());
  ASSERT_EQ(4, btd->end() - btd->begin());
  EXPECT_EQ('\xF0', btd->begin()[0]);
  EXPECT_EQ('\xA4', btd->begin()[1]);
  EXPECT_EQ('\xAD', btd->begin()[2]);
  EXPECT_EQ('\xA2', btd->begin()[3]);
}

TEST(View, NewAxis)
{
  nd::array a;

  a = parse_json("int", "1");
  EXPECT_JSON_EQ_ARR("[1]", a.new_axis(0));

  a = parse_json("5 * int", "[0, 1, 2, 3, 4]");
  EXPECT_JSON_EQ_ARR("[[0, 1, 2, 3, 4]]", a.new_axis(0));
  EXPECT_JSON_EQ_ARR("[[0], [1], [2], [3], [4]]", a.new_axis(1));

  a = parse_json("4 * string", "[\"dynd\", \"string\", \"hello\", \"world\"]");
  EXPECT_JSON_EQ_ARR("[[\"dynd\", \"string\", \"hello\", \"world\"]]", a.new_axis(0));
  EXPECT_JSON_EQ_ARR("[[\"dynd\"], [\"string\"], [\"hello\"], [\"world\"]]", a.new_axis(1));

  a = parse_json("2 * 3 * int", "[[0, 1, 2], [3, 4, 5]]");
  EXPECT_JSON_EQ_ARR("[[[0, 1, 2], [3, 4, 5]]]", a.new_axis(0));
  EXPECT_JSON_EQ_ARR("[[[0, 1, 2]], [[3, 4, 5]]]", a.new_axis(1));
  EXPECT_JSON_EQ_ARR("[[[0], [1], [2]], [[3], [4], [5]]]", a.new_axis(2));

  a = parse_json("4 * 3 * 2 * int", "[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]],"
                                    "[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]");
  EXPECT_JSON_EQ_ARR("[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]],"
                     "[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]",
                     a.new_axis(0));
  EXPECT_JSON_EQ_ARR("[[[[0, 1], [2, 3], [4, 5]]], [[[6, 7], [8, 9], [10, 11]]],"
                     "[[[12, 13], [14, 15], [16, 17]]], [[[18, 19], [20, 21], [22, 23]]]]",
                     a.new_axis(1));
  EXPECT_JSON_EQ_ARR("[[[[0, 1]], [[2, 3]], [[4, 5]]], [[[6, 7]], [[8, 9]], [[10, 11]]],"
                     "[[[12, 13]], [[14, 15]], [[16, 17]]], [[[18, 19]], [[20, 21]], [[22, 23]]]]",
                     a.new_axis(2));
  EXPECT_JSON_EQ_ARR("[[[[0], [1]], [[2], [3]], [[4], [5]]], [[[6], [7]], [[8], [9]], [[10], [11]]],"
                     "[[[12], [13]], [[14], [15]], [[16], [17]]], [[[18], [19]], [[20], [21]], [[22], [23]]]]",
                     a.new_axis(3));
}
