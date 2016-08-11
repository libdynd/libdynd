//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/array.hpp>
#include <dynd/assignment.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/gtest.hpp>

using namespace std;
using namespace dynd;

TEST(FixedDimType, Create) {
  ndt::type d;
  const ndt::fixed_dim_type *fad;

  // Strings with various encodings and sizes
  d = ndt::make_fixed_dim(3, ndt::make_type<int32_t>());
  EXPECT_EQ(fixed_dim_id, d.get_id());
  EXPECT_EQ(fixed_dim_kind_id, d.get_base_id());
  EXPECT_EQ(4u, d.get_data_alignment());
  EXPECT_EQ(0u, d.get_data_size());
  EXPECT_EQ(1, d.get_ndim());
  EXPECT_EQ(1, d.get_strided_ndim());
  EXPECT_FALSE(d.is_expression());
  EXPECT_EQ(ndt::make_type<int32_t>(), d.p<ndt::type>("element_type"));
  EXPECT_EQ(ndt::make_type<int32_t>(), d.at(-3));
  EXPECT_EQ(ndt::make_type<int32_t>(), d.at(-2));
  EXPECT_EQ(ndt::make_type<int32_t>(), d.at(-1));
  EXPECT_EQ(ndt::make_type<int32_t>(), d.at(0));
  EXPECT_EQ(ndt::make_type<int32_t>(), d.at(1));
  EXPECT_EQ(ndt::make_type<int32_t>(), d.at(2));
  fad = d.extended<ndt::fixed_dim_type>();
  EXPECT_EQ(3, fad->get_fixed_dim_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::make_fixed_dim(1, ndt::make_type<int32_t>());
  EXPECT_EQ(fixed_dim_id, d.get_id());
  EXPECT_EQ(fixed_dim_kind_id, d.get_base_id());
  EXPECT_EQ(4u, d.get_data_alignment());
  EXPECT_EQ(0u, d.get_data_size());
  EXPECT_FALSE(d.is_expression());
  fad = d.extended<ndt::fixed_dim_type>();
  EXPECT_EQ(1, fad->get_fixed_dim_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // Type constructor vs syntax sugar
  EXPECT_EQ(ndt::type("fixed[3] * int32"), ndt::type("3 * int32"));
}

TEST(FixedDimType, Basic) {
  nd::array a;
  float vals[3] = {1.5f, 2.5f, -1.5f};

  a = nd::empty(ndt::make_fixed_dim(3, ndt::make_type<float>()));
  a.vals() = vals;

  EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_type<float>()), a.get_type());
  EXPECT_EQ(1u, a.get_shape().size());
  EXPECT_EQ(3, a.get_shape()[0]);
  EXPECT_EQ(1u, a.get_strides().size());
  EXPECT_EQ(4, a.get_strides()[0]);
  EXPECT_EQ(1.5f, a(-3).as<float>());
  EXPECT_EQ(2.5f, a(-2).as<float>());
  EXPECT_EQ(-1.5f, a(-1).as<float>());
  EXPECT_EQ(1.5f, a(0).as<float>());
  EXPECT_EQ(2.5f, a(1).as<float>());
  EXPECT_EQ(-1.5f, a(2).as<float>());
  EXPECT_THROW(a(-4), index_out_of_bounds);
  EXPECT_THROW(a(3), index_out_of_bounds);
}

TEST(FixedDimType, SimpleIndex) {
  nd::array a = parse_json("2 * 3 * int16", "[[1, 2, 3], [4, 5, 6]]");
  ASSERT_EQ(ndt::make_fixed_dim(2, ndt::make_fixed_dim(3, ndt::make_type<int16_t>())), a.get_type());

  nd::array b;

  b = a(0);
  ASSERT_EQ(ndt::make_fixed_dim(3, ndt::make_type<int16_t>()), b.get_type());
  EXPECT_EQ(1, b(0).as<int16_t>());
  EXPECT_EQ(2, b(1).as<int16_t>());
  EXPECT_EQ(3, b(2).as<int16_t>());

  b = a(1);
  ASSERT_EQ(ndt::make_fixed_dim(3, ndt::make_type<int16_t>()), b.get_type());
  EXPECT_EQ(4, b(0).as<int16_t>());
  EXPECT_EQ(5, b(1).as<int16_t>());
  EXPECT_EQ(6, b(2).as<int16_t>());

  EXPECT_THROW(a(2), index_out_of_bounds);
  EXPECT_THROW(a(-3), index_out_of_bounds);
}

TEST(FixedDimType, AssignKernel_ScalarToFixed) {
  nd::array a, b;
  nd::kernel_builder k;

  // Assignment scalar -> fixed array
  a = nd::empty(ndt::make_fixed_dim(3, ndt::make_type<int>()));
  a.vals() = 0;
  b = 9.0;
  EXPECT_EQ(fixed_dim_id, a.get_type().get_id());
  EXPECT_ARRAY_EQ(nd::array({9, 9, 9}), a.assign(b));
}

TEST(FixedDimType, AssignKernel_FixedToFixed) {
  nd::array a, b;
  nd::kernel_builder k;

  // Assignment fixed array -> fixed array
  a = nd::empty(ndt::make_fixed_dim(3, ndt::make_type<int>()));
  a.vals() = 0;
  b = parse_json("3 * int32", "[3, 5, 7]");
  EXPECT_EQ(fixed_dim_id, a.get_type().get_id());
  EXPECT_EQ(fixed_dim_id, b.get_type().get_id());
  EXPECT_ARRAY_EQ(nd::array({3, 5, 7}), a.assign(b));
}

TEST(FixedDimType, AssignKernel_FixedToScalarError) {
  nd::array a, b;
  nd::kernel_builder k;

  // Assignment fixed array -> scalar
  a = 9.0;
  b = parse_json("3 * int32", "[3, 5, 7]");
  EXPECT_EQ(fixed_dim_id, b.get_type().get_id());
  EXPECT_THROW(a.assign(b), dynd::broadcast_error);
}

TEST(FixedDimType, IsTypeSubarray) {
  EXPECT_TRUE(ndt::type("3 * int32").is_type_subarray(ndt::type("3 * int32")));
  EXPECT_TRUE(ndt::type("10 * int32").is_type_subarray(ndt::type("10 * int32")));
  EXPECT_TRUE(ndt::type("3 * 10 * int32").is_type_subarray(ndt::type("10 * int32")));
  EXPECT_TRUE(ndt::type("3 * 10 * int32").is_type_subarray(ndt::type("int32")));
  EXPECT_TRUE(ndt::type("5 * int32").is_type_subarray(ndt::make_type<int32_t>()));
  EXPECT_FALSE(ndt::make_type<int32_t>().is_type_subarray(ndt::type("5 * int32")));
  EXPECT_FALSE(ndt::type("10 * int32").is_type_subarray(ndt::type("3 * 10 * int32")));
  EXPECT_FALSE(ndt::type("3 * int32").is_type_subarray(ndt::type("Fixed * int32")));
  EXPECT_FALSE(ndt::type("3 * int32").is_type_subarray(ndt::type("var * int32")));
  EXPECT_FALSE(ndt::type("Fixed * int32").is_type_subarray(ndt::type("3 * int32")));
  EXPECT_FALSE(ndt::type("var * int32").is_type_subarray(ndt::type("3 * int32")));
}

TEST(FixedDimType, FromCArray) {
  EXPECT_EQ(ndt::make_type<int>(), ndt::type("int32"));
  EXPECT_EQ(ndt::make_type<int[1]>(), ndt::type("1 * int32"));
  EXPECT_EQ(ndt::make_type<int[2]>(), ndt::type("2 * int32"));
  EXPECT_EQ(ndt::make_type<int[3]>(), ndt::type("3 * int32"));
  EXPECT_EQ(ndt::make_type<int[2][1]>(), ndt::type("2 * 1 * int32"));
  EXPECT_EQ(ndt::make_type<int[1][2]>(), ndt::type("1 * 2 * int32"));
  EXPECT_EQ(ndt::make_type<int[3][3]>(), ndt::type("3 * 3 * int32"));
  EXPECT_EQ(ndt::make_type<int[3][5][8][10]>(), ndt::type("3 * 5 * 8 * 10 * int32"));

  EXPECT_EQ(ndt::make_type<float>(), ndt::type("float32"));
  EXPECT_EQ(ndt::make_type<float[1]>(), ndt::type("1 * float32"));
  EXPECT_EQ(ndt::make_type<float[2]>(), ndt::type("2 * float32"));
  EXPECT_EQ(ndt::make_type<float[3]>(), ndt::type("3 * float32"));
  EXPECT_EQ(ndt::make_type<float[2][1]>(), ndt::type("2 * 1 * float32"));
  EXPECT_EQ(ndt::make_type<float[1][2]>(), ndt::type("1 * 2 * float32"));
  EXPECT_EQ(ndt::make_type<float[3][3]>(), ndt::type("3 * 3 * float32"));
  EXPECT_EQ(ndt::make_type<float[3][5][8][10]>(), ndt::type("3 * 5 * 8 * 10 * float32"));
}
