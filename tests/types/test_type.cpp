//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"
#include <complex>
#include <iostream>
#include <stdexcept>

#include <dynd/array.hpp>
#include <dynd/type.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

using namespace std;
using namespace dynd;

TEST(Type, NDTTypeObject) {
  // The ndt::type just contains one ndt::base_type *
  EXPECT_EQ(sizeof(ndt::base_type *), sizeof(ndt::type));
}

TEST(Type, BasicConstructor) {
  ndt::type d;

  // Default-constructed type properties
  EXPECT_EQ(uninitialized_id, d.get_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(0u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());

  // void type
  d = ndt::type(void_id);
  EXPECT_EQ(void_id, d.get_id());
  EXPECT_EQ(any_kind_id, d.get_base_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(0u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // bool type
  d = ndt::type(bool_id);
  EXPECT_EQ(bool_id, d.get_id());
  EXPECT_EQ(bool_kind_id, d.get_base_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(1u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // int8 type
  d = ndt::type(int8_id);
  EXPECT_EQ(int8_id, d.get_id());
  EXPECT_EQ(int_kind_id, d.get_base_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(1u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // int16 type
  d = ndt::type(int16_id);
  EXPECT_EQ(int_kind_id, d.get_base_id());
  EXPECT_EQ(2u, d.get_data_alignment());
  EXPECT_EQ(2u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // int32 type
  d = ndt::type(int32_id);
  EXPECT_EQ(int32_id, d.get_id());
  EXPECT_EQ(int_kind_id, d.get_base_id());
  EXPECT_EQ(4u, d.get_data_alignment());
  EXPECT_EQ(4u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // int
  d = ndt::make_type<int>();
  EXPECT_EQ(int32_id, d.get_id());
  EXPECT_EQ(int_kind_id, d.get_base_id());
  EXPECT_EQ(sizeof(int), d.get_data_alignment());
  EXPECT_EQ(sizeof(int), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // long
  d = ndt::make_type<long>();
  EXPECT_EQ(int_kind_id, d.get_base_id());
  EXPECT_EQ(sizeof(long), d.get_data_alignment());
  EXPECT_EQ(sizeof(long), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // long long
  d = ndt::make_type<long long>();
  EXPECT_EQ(int64_id, d.get_id());
  EXPECT_EQ(int_kind_id, d.get_base_id());
  EXPECT_EQ((size_t)alignof(long long), d.get_data_alignment());
  EXPECT_EQ(sizeof(long long), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // unsigned int
  d = ndt::make_type<unsigned int>();
  EXPECT_EQ(uint32_id, d.get_id());
  EXPECT_EQ(uint_kind_id, d.get_base_id());
  EXPECT_EQ(sizeof(unsigned int), d.get_data_alignment());
  EXPECT_EQ(sizeof(unsigned int), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // unsigned long
  d = ndt::make_type<unsigned long>();
  EXPECT_EQ(uint_kind_id, d.get_base_id());
  EXPECT_EQ(sizeof(unsigned long), d.get_data_alignment());
  EXPECT_EQ(sizeof(unsigned long), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // unsigned long long
  d = ndt::make_type<unsigned long long>();
  EXPECT_EQ(uint64_id, d.get_id());
  EXPECT_EQ(uint_kind_id, d.get_base_id());
  EXPECT_EQ((size_t)alignof(unsigned long long), d.get_data_alignment());
  EXPECT_EQ(sizeof(unsigned long long), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // float
  d = ndt::make_type<float>();
  EXPECT_EQ(float32_id, d.get_id());
  EXPECT_EQ(float_kind_id, d.get_base_id());
  EXPECT_EQ(sizeof(float), d.get_data_alignment());
  EXPECT_EQ(sizeof(float), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // double
  d = ndt::make_type<double>();
  EXPECT_EQ(float64_id, d.get_id());
  EXPECT_EQ(float_kind_id, d.get_base_id());
  EXPECT_EQ((size_t)alignof(double), d.get_data_alignment());
  EXPECT_EQ(sizeof(double), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(Type, TypeIDConstructor) {
  EXPECT_EQ(bool_id, ndt::type(bool_id).get_id());
  EXPECT_EQ(int8_id, ndt::type(int8_id).get_id());
  EXPECT_EQ(int16_id, ndt::type(int16_id).get_id());
  EXPECT_EQ(int32_id, ndt::type(int32_id).get_id());
  EXPECT_EQ(int64_id, ndt::type(int64_id).get_id());
  EXPECT_EQ(int128_id, ndt::type(int128_id).get_id());
  EXPECT_EQ(uint8_id, ndt::type(uint8_id).get_id());
  EXPECT_EQ(uint16_id, ndt::type(uint16_id).get_id());
  EXPECT_EQ(uint32_id, ndt::type(uint32_id).get_id());
  EXPECT_EQ(uint64_id, ndt::type(uint64_id).get_id());
  EXPECT_EQ(uint128_id, ndt::type(uint128_id).get_id());
  EXPECT_EQ(float16_id, ndt::type(float16_id).get_id());
  EXPECT_EQ(float32_id, ndt::type(float32_id).get_id());
  EXPECT_EQ(float64_id, ndt::type(float64_id).get_id());
  EXPECT_EQ(float128_id, ndt::type(float128_id).get_id());
  EXPECT_EQ(complex_float32_id, ndt::type(complex_float32_id).get_id());
  EXPECT_EQ(complex_float64_id, ndt::type(complex_float64_id).get_id());
  EXPECT_EQ(void_id, ndt::type(void_id).get_id());
  EXPECT_EQ(ndt::make_type<ndt::bytes_type>(), ndt::type(bytes_id));
  EXPECT_EQ(ndt::make_type<ndt::fixed_bytes_kind_type>(), ndt::type(fixed_bytes_id));
  EXPECT_EQ(ndt::make_type<ndt::pointer_type>(ndt::make_type<ndt::any_kind_type>()), ndt::type(pointer_id));
}

TEST(TypeFor, InitializerList) {
  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(1, ndt::make_type<int>()), ndt::type_for({0}));
  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<int>()), ndt::type_for({10, -2}));
  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(7, ndt::make_type<int>()), ndt::type_for({0, 1, 2, 3, 4, 5, 6}));

  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<int>())),
            ndt::type_for({{0, 1}, {2, 3}}));
  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<int>())),
            ndt::type_for({{0, 1, 2}, {3, 4, 5}}));

  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<ndt::var_dim_type>(ndt::make_type<int>())),
            ndt::type_for({{0}, {1, 2}}));
  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<ndt::var_dim_type>(ndt::make_type<int>())),
            ndt::type_for({{0, 1}, {2}}));
}
