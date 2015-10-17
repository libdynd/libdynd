//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/datashape_formatter.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixed_string_type.hpp>

using namespace std;
using namespace dynd;

TEST(DataShapeFormatter, ArrayBuiltinAtoms)
{
  // A NULL array
  EXPECT_EQ("uninitialized", format_datashape(nd::array()));
  // Scalar arrays of builtin types
  EXPECT_EQ("bool", format_datashape(nd::array(true), "", false));
  EXPECT_EQ("int8", format_datashape(nd::array((int8)0), "", false));
  EXPECT_EQ("int16", format_datashape(nd::array((int16)0), "", false));
  EXPECT_EQ("int32", format_datashape(nd::array((int32)0), "", false));
  EXPECT_EQ("int64", format_datashape(nd::array((int64)0), "", false));
  EXPECT_EQ("int128", format_datashape(nd::array(int128(0)), "", false));
  EXPECT_EQ("uint8", format_datashape(nd::array((uint8)0), "", false));
  EXPECT_EQ("uint16", format_datashape(nd::array((uint16)0), "", false));
  EXPECT_EQ("uint32", format_datashape(nd::array((uint32)0), "", false));
  EXPECT_EQ("uint64", format_datashape(nd::array((uint64)0), "", false));
  EXPECT_EQ("uint128", format_datashape(nd::array(uint128(0)), "", false));
  EXPECT_EQ("float16", format_datashape(nd::array(float16(0.f)), "", false));
  EXPECT_EQ("float32", format_datashape(nd::array(0.f), "", false));
  EXPECT_EQ("float64", format_datashape(nd::array(0.), "", false));
  EXPECT_EQ("complex[float32]", format_datashape(nd::array(dynd::complex<float>(0.f)), "", false));
  EXPECT_EQ("complex[float64]", format_datashape(nd::array(dynd::complex<double>(0.)), "", false));
}

TEST(DataShapeFormatter, DTypeBuiltinAtoms)
{
  EXPECT_EQ("bool", format_datashape(ndt::type::make<bool1>(), "", false));
  EXPECT_EQ("int8", format_datashape(ndt::type::make<int8_t>(), "", false));
  EXPECT_EQ("int16", format_datashape(ndt::type::make<int16_t>(), "", false));
  EXPECT_EQ("int32", format_datashape(ndt::type::make<int32_t>(), "", false));
  EXPECT_EQ("int64", format_datashape(ndt::type::make<int64_t>(), "", false));
  EXPECT_EQ("uint8", format_datashape(ndt::type::make<uint8_t>(), "", false));
  EXPECT_EQ("uint16", format_datashape(ndt::type::make<uint16_t>(), "", false));
  EXPECT_EQ("uint32", format_datashape(ndt::type::make<uint32_t>(), "", false));
  EXPECT_EQ("uint64", format_datashape(ndt::type::make<uint64_t>(), "", false));
  EXPECT_EQ("float32", format_datashape(ndt::type::make<float>(), "", false));
  EXPECT_EQ("float64", format_datashape(ndt::type::make<double>(), "", false));
  EXPECT_EQ("complex[float32]", format_datashape(ndt::type::make<dynd::complex<float>>(), "", false));
  EXPECT_EQ("complex[float64]", format_datashape(ndt::type::make<dynd::complex<double>>(), "", false));
}

TEST(DataShapeFormatter, ArrayStringAtoms)
{
  EXPECT_EQ("string", format_datashape(nd::array("test"), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::string_type::make()), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::string_type::make()), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::string_type::make()), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::string_type::make()), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::string_type::make()), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::fixed_string_type::make(1, string_encoding_utf_8)), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::fixed_string_type::make(10, string_encoding_utf_8)), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::fixed_string_type::make(10, string_encoding_ascii)), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::fixed_string_type::make(10, string_encoding_utf_16)), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::fixed_string_type::make(10, string_encoding_utf_32)), "", false));
  EXPECT_EQ("string", format_datashape(nd::empty(ndt::fixed_string_type::make(10, string_encoding_ucs_2)), "", false));
}

TEST(DataShapeFormatter, DTypeStringAtoms)
{
  EXPECT_EQ("string", format_datashape(ndt::string_type::make(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::string_type::make(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::string_type::make(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::string_type::make(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::string_type::make(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::string_type::make(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(1, string_encoding_utf_8), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(10, string_encoding_utf_8), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(10, string_encoding_ascii), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(10, string_encoding_utf_16), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(10, string_encoding_utf_32), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(10, string_encoding_ucs_2), "", false));
}

TEST(DataShapeFormatter, ArrayUniformArrays)
{
  EXPECT_EQ("3 * int32", format_datashape(nd::empty<int32_t[3]>(), "", false));
  EXPECT_EQ("var * int32", format_datashape(nd::empty(ndt::var_dim_type::make(ndt::type::make<int32_t>())), "", false));
  EXPECT_EQ("var * 3 * int32",
            format_datashape(nd::empty(ndt::var_dim_type::make(ndt::make_fixed_dim(3, ndt::type::make<int32_t>()))), "",
                             false));
}

TEST(DataShapeFormatter, DTypeUniformArrays)
{
  EXPECT_EQ("Fixed * Fixed * Fixed * int32",
            format_datashape(ndt::make_fixed_dim_kind(ndt::type::make<int32_t>(), 3), "", false));
  EXPECT_EQ("var * int32", format_datashape(ndt::var_dim_type::make(ndt::type::make<int32_t>()), "", false));
  EXPECT_EQ("var * 3 * int32",
            format_datashape(ndt::var_dim_type::make(ndt::make_fixed_dim(3, ndt::type::make<int32_t>())), "", false));
  EXPECT_EQ("var * Fixed * int32",
            format_datashape(ndt::var_dim_type::make(ndt::make_fixed_dim_kind(ndt::type::make<int32_t>())), "", false));
}

TEST(DataShapeFormatter, ArrayStructs)
{
  EXPECT_EQ("{x: int32, y: float64}",
            format_datashape(
                nd::empty(ndt::struct_type::make({"x", "y"}, {ndt::type::make<int32_t>(), ndt::type::make<double>()})),
                "", false));
  EXPECT_EQ(
      "{x: var * {a: int32, b: int8}, y: 5 * var * uint8}",
      format_datashape(nd::empty(ndt::struct_type::make(
                           {"x", "y"}, {ndt::var_dim_type::make(ndt::struct_type::make(
                                            {"a", "b"}, {ndt::type::make<int32_t>(), ndt::type::make<int8_t>()})),
                                        ndt::make_fixed_dim(5, ndt::var_dim_type::make(ndt::type::make<uint8_t>()))})),
                       "", false));
}

TEST(DataShapeFormatter, DTypeStructs)
{
  EXPECT_EQ("{x: int32, y: float64}", format_datashape(ndt::struct_type::make({"x", "y"}, {ndt::type::make<int32_t>(),
                                                                                           ndt::type::make<double>()}),
                                                       "", false));
  EXPECT_EQ(
      "{x: var * {a: int32, b: int8}, y: 5 * var * uint8}",
      format_datashape(ndt::struct_type::make(
                           {"x", "y"}, {ndt::var_dim_type::make(ndt::struct_type::make(
                                            {"a", "b"}, {ndt::type::make<int32_t>(), ndt::type::make<int8_t>()})),
                                        ndt::make_fixed_dim(5, ndt::var_dim_type::make(ndt::type::make<uint8_t>()))}),
                       "", false));
  EXPECT_EQ(
      "{x: 7 * {a: int32, b: int8}, y: var * 4 * uint8}",
      format_datashape(
          ndt::struct_type::make(
              {"x", "y"}, {ndt::make_fixed_dim(7, ndt::struct_type::make({"a", "b"}, {ndt::type::make<int32_t>(),
                                                                                      ndt::type::make<int8_t>()})),
                           ndt::var_dim_type::make(ndt::make_fixed_dim(4, ndt::type::make<uint8_t>()))}),
          "", false));
}
