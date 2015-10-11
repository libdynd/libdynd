//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/byteswap_type.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

TEST(StructType, Basic)
{
  EXPECT_NE(ndt::struct_type::make({"x"}, {ndt::type::make<int>()}),
            ndt::struct_type::make({"y"}, {ndt::type::make<int>()}));
  EXPECT_NE(ndt::struct_type::make({"x"}, {ndt::type::make<float>()}),
            ndt::struct_type::make({"x"}, {ndt::type::make<int>()}));
}

TEST(StructType, Equality)
{
  EXPECT_EQ(ndt::type("{x: int32, y: float16, z: int32}"), ndt::type("{x: int32, y: float16, z: int32}"));
  EXPECT_NE(ndt::type("{x: int32, y: float16, z: int32}"), ndt::type("{x: int32, y: float16, z: int32, ...}"));
}

TEST(StructType, IOStream)
{
  stringstream ss;
  ndt::type tp;

  tp = ndt::struct_type::make({"x"}, {ndt::type::make<float>()});
  ss << tp;
  EXPECT_EQ("{x: float32}", ss.str());

  ss.str("");
  ss.clear();
  tp = ndt::struct_type::make({"x", "y"}, {ndt::type::make<int32_t>(), ndt::type::make<int16_t>()});
  ss << tp;
  EXPECT_EQ("{x: int32, y: int16}", ss.str());

  ss.str("");
  ss.clear();
  tp = ndt::struct_type::make({"x", "y", "Verbose Field!"},
                              {ndt::type::make<int32_t>(), ndt::type::make<int16_t>(), ndt::type::make<float>()});
  ss << tp;
  EXPECT_EQ("{x: int32, y: int16, 'Verbose Field!': float32}", ss.str());
}

TEST(StructType, CreateOneField)
{
  ndt::type dt;
  const ndt::struct_type *tdt;

  // Struct with one field
  dt = ndt::struct_type::make({"x"}, {ndt::type::make<int32_t>()});
  EXPECT_EQ(struct_type_id, dt.get_type_id());
  EXPECT_EQ(struct_kind, dt.get_kind());
  EXPECT_EQ(0u, dt.get_data_size()); // No size
  EXPECT_EQ(4u, dt.extended()->get_default_data_size());
  EXPECT_EQ(4u, dt.get_data_alignment());
  EXPECT_FALSE(dt.is_pod());
  EXPECT_EQ(0u, (dt.get_flags() & (type_flag_blockref | type_flag_destructor)));
  tdt = dt.extended<ndt::struct_type>();
  EXPECT_EQ(1, tdt->get_field_count());
  EXPECT_EQ(ndt::type::make<int32_t>(), tdt->get_field_type(0));
  EXPECT_EQ("x", tdt->get_field_name(0));
}

struct two_field_struct {
  int64_t a;
  int32_t b;
};

TEST(StructType, CreateTwoField)
{
  ndt::type dt;
  const ndt::struct_type *tdt;

  // Struct with two fields
  dt = ndt::struct_type::make({"a", "b"}, {ndt::type::make<int64_t>(), ndt::type::make<int32_t>()});
  EXPECT_EQ(struct_type_id, dt.get_type_id());
  EXPECT_EQ(struct_kind, dt.get_kind());
  EXPECT_EQ(0u, dt.get_data_size());
  EXPECT_EQ(sizeof(two_field_struct), dt.extended()->get_default_data_size());
  EXPECT_EQ((size_t)scalar_align_of<two_field_struct>::value, dt.get_data_alignment());
  EXPECT_FALSE(dt.is_pod());
  EXPECT_EQ(0u, (dt.get_flags() & (type_flag_blockref | type_flag_destructor)));
  tdt = dt.extended<ndt::struct_type>();
  EXPECT_EQ(2, tdt->get_field_count());
  EXPECT_EQ(ndt::type::make<int64_t>(), tdt->get_field_type(0));
  EXPECT_EQ(ndt::type::make<int32_t>(), tdt->get_field_type(1));
  EXPECT_EQ("a", tdt->get_field_name(0));
  EXPECT_EQ("b", tdt->get_field_name(1));
}

struct three_field_struct {
  int64_t x;
  int32_t y;
  char z[5];
};

TEST(StructType, CreateThreeField)
{
  ndt::type dt;
  const ndt::struct_type *tdt;

  // Struct with three fields
  ndt::type d1 = ndt::type::make<int64_t>();
  ndt::type d2 = ndt::type::make<int32_t>();
  ndt::type d3 = ndt::fixed_string_type::make(5, string_encoding_utf_8);
  dt = ndt::struct_type::make({"x", "y", "z"}, {d1, d2, d3});
  EXPECT_EQ(struct_type_id, dt.get_type_id());
  EXPECT_EQ(struct_kind, dt.get_kind());
  EXPECT_EQ(0u, dt.get_data_size());
  EXPECT_EQ(sizeof(three_field_struct), dt.extended()->get_default_data_size());
  EXPECT_EQ((size_t)scalar_align_of<two_field_struct>::value, dt.get_data_alignment());
  EXPECT_FALSE(dt.is_pod());
  EXPECT_EQ(0u, (dt.get_flags() & (type_flag_blockref | type_flag_destructor)));
  tdt = dt.extended<ndt::struct_type>();
  EXPECT_EQ(3, tdt->get_field_count());
  EXPECT_EQ(ndt::type::make<int64_t>(), tdt->get_field_type(0));
  EXPECT_EQ(ndt::type::make<int32_t>(), tdt->get_field_type(1));
  EXPECT_EQ(ndt::fixed_string_type::make(5, string_encoding_utf_8), tdt->get_field_type(2));
  EXPECT_EQ("x", tdt->get_field_name(0));
  EXPECT_EQ("y", tdt->get_field_name(1));
  EXPECT_EQ("z", tdt->get_field_name(2));
}

TEST(StructType, ReplaceScalarTypes)
{
  ndt::type dt, dt2;

  // Struct with three fields
  ndt::type d1 = ndt::type::make<dynd::complex<double>>();
  ndt::type d2 = ndt::type::make<int32_t>();
  ndt::type d3 = ndt::fixed_string_type::make(5, string_encoding_utf_8);
  dt = ndt::struct_type::make({"x", "y", "z"}, {d1, d2, d3});
  dt2 = dt.with_replaced_scalar_types(ndt::type::make<int16_t>());
  EXPECT_EQ(ndt::struct_type::make({"x", "y", "z"}, {ndt::convert_type::make(ndt::type::make<int16_t>(), d1),
                                                     ndt::convert_type::make(ndt::type::make<int16_t>(), d2),
                                                     ndt::convert_type::make(ndt::type::make<int16_t>(), d3)}),
            dt2);
}

TEST(StructType, TypeAt)
{
  ndt::type dt, dt2;

  // Struct with three fields
  ndt::type d1 = ndt::type::make<dynd::complex<double>>();
  ndt::type d2 = ndt::type::make<int32_t>();
  ndt::type d3 = ndt::fixed_string_type::make(5, string_encoding_utf_8);
  dt = ndt::struct_type::make({"x", "y", "z"}, {d1, d2, d3});

  // indexing into a type with a slice produces another
  // struct type with the subset of fields.
  EXPECT_EQ(ndt::struct_type::make({"x", "y"}, {d1, d2}), dt.at(irange() < 2));
  EXPECT_EQ(ndt::struct_type::make({"x", "z"}, {d1, d3}), dt.at(irange(0, 3, 2)));
  EXPECT_EQ(ndt::struct_type::make({"z", "y"}, {d3, d2}), dt.at(irange(2, 0, -1)));
}

TEST(StructType, CanonicalType)
{
  ndt::type dt, dt2;

  // Struct with three fields
  ndt::type d1 = ndt::convert_type::make(ndt::type::make<dynd::complex<double>>(), ndt::type::make<float>());
  ndt::type d2 = ndt::byteswap_type::make(ndt::type::make<int32_t>());
  ndt::type d3 = ndt::fixed_string_type::make(5, string_encoding_utf_32);
  dt = ndt::struct_type::make({"x", "y", "z"}, {d1, d2, d3});
  EXPECT_EQ(ndt::struct_type::make({"x", "y", "z"},
                                   {ndt::type::make<dynd::complex<double>>(), ndt::type::make<int32_t>(), d3}),
            dt.get_canonical_type());
}

TEST(StructType, IsExpression)
{
  ndt::type d1 = ndt::type::make<float>();
  ndt::type d2 = ndt::byteswap_type::make(ndt::type::make<int32_t>());
  ndt::type d3 = ndt::fixed_string_type::make(5, string_encoding_utf_32);
  ndt::type d = ndt::struct_type::make({"x", "y", "z"}, {d1, d2, d3});

  EXPECT_TRUE(d.is_expression());
  EXPECT_FALSE(d.at(irange(0, 3, 2)).is_expression());
}

TEST(StructType, PropertyAccess)
{
  ndt::type dt = ndt::struct_type::make({"x", "y", "z"},
                                        {ndt::type::make<int>(), ndt::type::make<double>(), ndt::type::make<short>()});
  nd::array a = nd::empty(dt);
  a(0).vals() = 3;
  a(1).vals() = 4.25;
  a(2).vals() = 5;
  EXPECT_EQ(3, a.p("x").as<int>());
  EXPECT_EQ(4.25, a.p("y").as<double>());
  EXPECT_EQ(5, a.p("z").as<short>());
  EXPECT_THROW(a.p("w"), runtime_error);
}

TEST(StructType, EqualTypeAssign)
{
  ndt::type dt = ndt::struct_type::make({"x", "y", "z"},
                                        {ndt::type::make<int>(), ndt::type::make<double>(), ndt::type::make<short>()});
  nd::array a = nd::empty(2, dt);
  a(0, 0).vals() = 3;
  a(0, 1).vals() = 4.25;
  a(0, 2).vals() = 5;
  a(1, 0).vals() = 6;
  a(1, 1).vals() = 7.25;
  a(1, 2).vals() = 8;

  nd::array b = nd::empty(2, dt);
  b.val_assign(a);
  EXPECT_EQ(3, a(0, 0).as<int>());
  EXPECT_EQ(4.25, a(0, 1).as<double>());
  EXPECT_EQ(5, a(0, 2).as<short>());
  EXPECT_EQ(6, a(1, 0).as<int>());
  EXPECT_EQ(7.25, a(1, 1).as<double>());
  EXPECT_EQ(8, a(1, 2).as<short>());
}

TEST(StructType, DifferentTypeAssign)
{
  ndt::type dt = ndt::struct_type::make({"x", "y", "z"},
                                        {ndt::type::make<int>(), ndt::type::make<double>(), ndt::type::make<short>()});
  nd::array a = nd::empty(2, dt);
  a(0, 0).vals() = 3;
  a(0, 1).vals() = 4.25;
  a(0, 2).vals() = 5;
  a(1, 0).vals() = 6;
  a(1, 1).vals() = 7.25;
  a(1, 2).vals() = 8;

  ndt::type dt2 = ndt::struct_type::make(
      {"y", "z", "x"}, {ndt::type::make<float>(), ndt::type::make<int>(), ndt::type::make<uint8_t>()});
  nd::array b = nd::empty(2, dt2);
  b.val_assign(a);
  EXPECT_EQ(3, b(0, 2).as<int>());
  EXPECT_EQ(4.25, b(0, 0).as<double>());
  EXPECT_EQ(5, b(0, 1).as<short>());
  EXPECT_EQ(6, b(1, 2).as<int>());
  EXPECT_EQ(7.25, b(1, 0).as<double>());
  EXPECT_EQ(8, b(1, 1).as<short>());
}

TEST(StructType, SingleCompare)
{
  nd::array a, b;
  ndt::type sdt = ndt::struct_type::make(
      {"a", "b", "c"}, {ndt::type::make<int32_t>(), ndt::type::make<float>(), ndt::type::make<int64_t>()});
  a = nd::empty(sdt);
  b = nd::empty(sdt);

  // Test lexicographic sorting

  // a == b
  a.p("a").vals() = 3;
  a.p("b").vals() = -2.25;
  a.p("c").vals() = 66;
  b.p("a").vals() = 3;
  b.p("b").vals() = -2.25;
  b.p("c").vals() = 66;
  // EXPECT_FALSE(a.op_sorting_less(b));
  // EXPECT_THROW((a < b), not_comparable_error);
  // EXPECT_THROW((a <= b), not_comparable_error);
  EXPECT_TRUE(static_cast<bool>(a == b));
  EXPECT_FALSE(static_cast<bool>(a != b));
  //  EXPECT_THROW((a >= b), not_comparable_error);
  //  EXPECT_THROW((a > b), not_comparable_error);
  //  EXPECT_FALSE(b.op_sorting_less(a));
  // EXPECT_THROW((b < a), not_comparable_error);
  // EXPECT_THROW((b <= a), not_comparable_error);
  EXPECT_TRUE(static_cast<bool>(b == a));
  EXPECT_FALSE(static_cast<bool>(b != a));
  // EXPECT_THROW((b >= a), not_comparable_error);
  // EXPECT_THROW((b > a), not_comparable_error);

  // Different in the first field
  a.p("a").vals() = 3;
  a.p("b").vals() = -2.25;
  a.p("c").vals() = 66;
  b.p("a").vals() = 4;
  b.p("b").vals() = -2.25;
  b.p("c").vals() = 66;
  //   EXPECT_TRUE(a.op_sorting_less(b));
  // EXPECT_THROW((a < b), not_comparable_error);
  // EXPECT_THROW((a <= b), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(a == b));
  EXPECT_TRUE(static_cast<bool>(a != b));
  //   EXPECT_THROW((a >= b), not_comparable_error);
  //  EXPECT_THROW((a > b), not_comparable_error);
  //    EXPECT_FALSE(b.op_sorting_less(a));
  // EXPECT_THROW((b < a), not_comparable_error);
  //  EXPECT_THROW((b <= a), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(b == a));
  EXPECT_TRUE(static_cast<bool>(b != a));
  //    EXPECT_THROW((b >= a), not_comparable_error);
  //      EXPECT_THROW((b > a), not_comparable_error);

  // Different in the second field
  a.p("a").vals() = 3;
  a.p("b").vals() = -2.25;
  a.p("c").vals() = 66;
  b.p("a").vals() = 3;
  b.p("b").vals() = -2.23;
  b.p("c").vals() = 66;
  //  EXPECT_TRUE(a.op_sorting_less(b));
  //      EXPECT_THROW((a < b), not_comparable_error);
  //    EXPECT_THROW((a <= b), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(a == b));
  EXPECT_TRUE(static_cast<bool>(a != b));
  //      EXPECT_THROW((a >= b), not_comparable_error);
  //    EXPECT_THROW((a > b), not_comparable_error);
  //      EXPECT_FALSE(b.op_sorting_less(a));
  //    EXPECT_THROW((b < a), not_comparable_error);
  //  EXPECT_THROW((b <= a), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(b == a));
  EXPECT_TRUE(static_cast<bool>(b != a));
  //    EXPECT_THROW((b >= a), not_comparable_error);
  //      EXPECT_THROW((b > a), not_comparable_error);

  // Different in the third field
  a.p("a").vals() = 3;
  a.p("b").vals() = -2.25;
  a.p("c").vals() = 66;
  b.p("a").vals() = 3;
  b.p("b").vals() = -2.25;
  b.p("c").vals() = 1000;
  //    EXPECT_TRUE(a.op_sorting_less(b));
  //  EXPECT_THROW((a < b), not_comparable_error);
  // EXPECT_THROW((a <= b), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(a == b));
  EXPECT_TRUE(static_cast<bool>(a != b));
  //   EXPECT_THROW((a >= b), not_comparable_error);
  //  EXPECT_THROW((a > b), not_comparable_error);
  //    EXPECT_FALSE(b.op_sorting_less(a));
  // EXPECT_THROW((b < a), not_comparable_error);
  //  EXPECT_THROW((b <= a), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(b == a));
  EXPECT_TRUE(static_cast<bool>(b != a));
  //    EXPECT_THROW((b >= a), not_comparable_error);
  //      EXPECT_THROW((b > a), not_comparable_error);
}

TEST(StructType, SingleCompareDifferentArrmeta)
{
  nd::array a, b;
  ndt::type sdt = ndt::struct_type::make(
      {"a", "b", "c"}, {ndt::type::make<int32_t>(), ndt::type::make<float>(), ndt::type::make<int64_t>()});
  ndt::type sdt_reverse = ndt::struct_type::make(
      {"c", "b", "a"}, {ndt::type::make<int64_t>(), ndt::type::make<float>(), ndt::type::make<int32_t>()});
  a = nd::empty(sdt);
  b = nd::empty(sdt_reverse)(irange().by(-1));

  // Confirm that the arrmeta is different
  EXPECT_EQ(a.get_type(), b.get_type());
  const ndt::struct_type *a_sdt = static_cast<const ndt::struct_type *>(a.get_type().extended());
  const ndt::struct_type *b_sdt = static_cast<const ndt::struct_type *>(b.get_type().extended());
  EXPECT_NE(a_sdt->get_data_offsets(a.get_arrmeta())[0], b_sdt->get_data_offsets(b.get_arrmeta())[0]);
  EXPECT_NE(a_sdt->get_data_offsets(a.get_arrmeta())[1], b_sdt->get_data_offsets(b.get_arrmeta())[1]);
  EXPECT_NE(a_sdt->get_data_offsets(a.get_arrmeta())[2], b_sdt->get_data_offsets(b.get_arrmeta())[2]);

  // Test lexicographic sorting

  // a == b
  a.p("a").vals() = 3;
  a.p("b").vals() = -2.25;
  a.p("c").vals() = 66;
  b.p("a").vals() = 3;
  b.p("b").vals() = -2.25;
  b.p("c").vals() = 66;
  // EXPECT_FALSE(a.op_sorting_less(b));
  //   EXPECT_THROW((a < b), not_comparable_error);
  // EXPECT_THROW((a <= b), not_comparable_error);
  EXPECT_TRUE(static_cast<bool>(a == b));
  EXPECT_FALSE(static_cast<bool>(a != b));
  //    EXPECT_THROW((a >= b), not_comparable_error);
  //  EXPECT_THROW((a > b), not_comparable_error);
  //   EXPECT_FALSE(b.op_sorting_less(a));
  //   EXPECT_THROW((b < a), not_comparable_error);
  // EXPECT_THROW((b <= a), not_comparable_error);
  EXPECT_TRUE(static_cast<bool>(b == a));
  EXPECT_FALSE(static_cast<bool>(b != a));
  //  EXPECT_THROW((b >= a), not_comparable_error);
  //    EXPECT_THROW((b > a), not_comparable_error);

  // Different in the first field
  a.p("a").vals() = 3;
  a.p("b").vals() = -2.25;
  a.p("c").vals() = 66;
  b.p("a").vals() = 4;
  b.p("b").vals() = -2.25;
  b.p("c").vals() = 66;
  //      EXPECT_TRUE(a.op_sorting_less(b));
  //    EXPECT_THROW((a < b), not_comparable_error);
  //  EXPECT_THROW((a <= b), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(a == b));
  EXPECT_TRUE(static_cast<bool>(a != b));
  // EXPECT_THROW((a >= b), not_comparable_error);
  //   EXPECT_THROW((a > b), not_comparable_error);
  // EXPECT_FALSE(b.op_sorting_less(a));
  //    EXPECT_THROW((b < a), not_comparable_error);
  //  EXPECT_THROW((b <= a), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(b == a));
  EXPECT_TRUE(static_cast<bool>(b != a));
  // EXPECT_THROW((b >= a), not_comparable_error);
  // EXPECT_THROW((b > a), not_comparable_error);

  // Different in the second field
  a.p("a").vals() = 3;
  a.p("b").vals() = -2.25;
  a.p("c").vals() = 66;
  b.p("a").vals() = 3;
  b.p("b").vals() = -2.23;
  b.p("c").vals() = 66;
  //      EXPECT_TRUE(a.op_sorting_less(b));
  //    EXPECT_THROW((a < b), not_comparable_error);
  //   EXPECT_THROW((a <= b), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(a == b));
  EXPECT_TRUE(static_cast<bool>(a != b));
  // EXPECT_THROW((a >= b), not_comparable_error);
  // EXPECT_THROW((a > b), not_comparable_error);
  //   EXPECT_FALSE(b.op_sorting_less(a));
  // EXPECT_THROW((b < a), not_comparable_error);
  //   EXPECT_THROW((b <= a), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(b == a));
  EXPECT_TRUE(static_cast<bool>(b != a));
  // EXPECT_THROW((b >= a), not_comparable_error);
  //  EXPECT_THROW((b > a), not_comparable_error);

  // Different in the third field
  a.p("a").vals() = 3;
  a.p("b").vals() = -2.25;
  a.p("c").vals() = 66;
  b.p("a").vals() = 3;
  b.p("b").vals() = -2.25;
  b.p("c").vals() = 1000;
  //      EXPECT_TRUE(a.op_sorting_less(b));
  //    EXPECT_THROW((a < b), not_comparable_error);
  //  EXPECT_THROW((a <= b), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(a == b));
  EXPECT_TRUE(static_cast<bool>(a != b));
  // EXPECT_THROW((a >= b), not_comparable_error);
  //    EXPECT_THROW((a > b), not_comparable_error);
  //   EXPECT_FALSE(b.op_sorting_less(a));
  //  EXPECT_THROW((b < a), not_comparable_error);
  // EXPECT_THROW((b <= a), not_comparable_error);
  EXPECT_FALSE(static_cast<bool>(b == a));
  EXPECT_TRUE(static_cast<bool>(b != a));
  //   EXPECT_THROW((b >= a), not_comparable_error);
  //     EXPECT_THROW((b > a), not_comparable_error);
}
