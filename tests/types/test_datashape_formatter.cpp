//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/datashape_formatter.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixed_string_type.hpp>

using namespace std;
using namespace dynd;

TEST(DataShapeFormatter, DTypeBuiltinAtoms) {
  EXPECT_EQ("bool", format_datashape(ndt::make_type<bool1>(), "", false));
  EXPECT_EQ("int8", format_datashape(ndt::make_type<int8_t>(), "", false));
  EXPECT_EQ("int16", format_datashape(ndt::make_type<int16_t>(), "", false));
  EXPECT_EQ("int32", format_datashape(ndt::make_type<int32_t>(), "", false));
  EXPECT_EQ("int64", format_datashape(ndt::make_type<int64_t>(), "", false));
  EXPECT_EQ("uint8", format_datashape(ndt::make_type<uint8_t>(), "", false));
  EXPECT_EQ("uint16", format_datashape(ndt::make_type<uint16_t>(), "", false));
  EXPECT_EQ("uint32", format_datashape(ndt::make_type<uint32_t>(), "", false));
  EXPECT_EQ("uint64", format_datashape(ndt::make_type<uint64_t>(), "", false));
  EXPECT_EQ("float32", format_datashape(ndt::make_type<float>(), "", false));
  EXPECT_EQ("float64", format_datashape(ndt::make_type<double>(), "", false));
  EXPECT_EQ("complex[float32]", format_datashape(ndt::make_type<dynd::complex<float>>(), "", false));
  EXPECT_EQ("complex[float64]", format_datashape(ndt::make_type<dynd::complex<double>>(), "", false));
}

TEST(DataShapeFormatter, DTypeStringAtoms) {
  EXPECT_EQ("string", format_datashape(ndt::make_type<ndt::string_type>(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::make_type<ndt::string_type>(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::make_type<ndt::string_type>(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::make_type<ndt::string_type>(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::make_type<ndt::string_type>(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::make_type<ndt::string_type>(), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(1, string_encoding_utf_8), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(10, string_encoding_utf_8), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(10, string_encoding_ascii), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(10, string_encoding_utf_16), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(10, string_encoding_utf_32), "", false));
  EXPECT_EQ("string", format_datashape(ndt::fixed_string_type::make(10, string_encoding_ucs_2), "", false));
}

TEST(DataShapeFormatter, DTypeUniformArrays) {
  EXPECT_EQ("Fixed * Fixed * Fixed * int32",
            format_datashape(ndt::make_fixed_dim_kind(ndt::make_type<int32_t>(), 3), "", false));
  EXPECT_EQ("var * int32", format_datashape(ndt::make_type<ndt::var_dim_type>(ndt::make_type<int32_t>()), "", false));
  EXPECT_EQ("var * 3 * int32",
            format_datashape(ndt::var_dim_type::make(ndt::make_fixed_dim(3, ndt::make_type<int32_t>())), "", false));
  EXPECT_EQ("var * Fixed * int32",
            format_datashape(ndt::var_dim_type::make(ndt::make_fixed_dim_kind(ndt::make_type<int32_t>())), "", false));
}

TEST(DataShapeFormatter, DTypeStructs) {
  EXPECT_EQ("{x: int32, y: float64}",
            format_datashape(
                ndt::make_type<ndt::struct_type>({{ndt::make_type<int32_t>(), "x"}, {ndt::make_type<double>(), "y"}}),
                "", false));
  EXPECT_EQ(
      "{x: var * {a: int32, b: int8}, y: 5 * var * uint8}",
      format_datashape(
          ndt::make_type<ndt::struct_type>(
              {{ndt::make_type<ndt::var_dim_type>(ndt::make_type<ndt::struct_type>(
                    {{ndt::make_type<int32_t>(), "a"}, {ndt::make_type<int8_t>(), "b"}})),
                "x"},
               {ndt::make_type<ndt::fixed_dim_type>(5, ndt::make_type<ndt::var_dim_type>(ndt::make_type<uint8_t>())),
                "y"}}),
          "", false));
  EXPECT_EQ("{x: 7 * {a: int32, b: int8}, y: var * 4 * uint8}",
            format_datashape(
                ndt::make_type<ndt::struct_type>(
                    {{ndt::make_fixed_dim(7, ndt::make_type<ndt::struct_type>(
                                                 {{ndt::make_type<int32_t>(), "a"}, {ndt::make_type<int8_t>(), "b"}})),
                      "x"},
                     {ndt::make_type<ndt::var_dim_type>(ndt::make_fixed_dim(4, ndt::make_type<uint8_t>())), "y"}}),
                "", false));
}
