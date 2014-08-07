//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/datashape_formatter.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/func/callable.hpp>

using namespace std;
using namespace dynd;

TEST(DataShapeFormatter, ArrayBuiltinAtoms) {
  // A NULL array
  EXPECT_EQ("uninitialized", format_datashape(nd::array()));
  // Scalar arrays of builtin types
  EXPECT_EQ("bool", format_datashape(nd::array(true), "", false));
  EXPECT_EQ("int8", format_datashape(nd::array((int8_t)0), "", false));
  EXPECT_EQ("int16", format_datashape(nd::array((int16_t)0), "", false));
  EXPECT_EQ("int32", format_datashape(nd::array((int32_t)0), "", false));
  EXPECT_EQ("int64", format_datashape(nd::array((int64_t)0), "", false));
  EXPECT_EQ("int128", format_datashape(nd::array(dynd_int128(0)), "", false));
  EXPECT_EQ("uint8", format_datashape(nd::array((uint8_t)0), "", false));
  EXPECT_EQ("uint16", format_datashape(nd::array((uint16_t)0), "", false));
  EXPECT_EQ("uint32", format_datashape(nd::array((uint32_t)0), "", false));
  EXPECT_EQ("uint64", format_datashape(nd::array((uint64_t)0), "", false));
  EXPECT_EQ("uint128", format_datashape(nd::array(dynd_uint128(0)), "", false));
  EXPECT_EQ("float16", format_datashape(nd::array(dynd_float16(0.f, assign_error_nocheck)), "", false));
  EXPECT_EQ("float32", format_datashape(nd::array(0.f), "", false));
  EXPECT_EQ("float64", format_datashape(nd::array(0.), "", false));
  EXPECT_EQ("complex[float32]", format_datashape(nd::array(dynd_complex<float>(0.f)), "", false));
  EXPECT_EQ("complex[float64]", format_datashape(nd::array(dynd_complex<double>(0.)), "", false));
}

TEST(DataShapeFormatter, DTypeBuiltinAtoms) {
    EXPECT_EQ("bool", format_datashape(ndt::make_type<dynd_bool>(), "", false));
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
    EXPECT_EQ("complex[float32]", format_datashape(ndt::make_type<dynd_complex<float> >(), "", false));
    EXPECT_EQ("complex[float64]", format_datashape(ndt::make_type<dynd_complex<double> >(), "", false));
}

TEST(DataShapeFormatter, ArrayStringAtoms) {
    EXPECT_EQ("string", format_datashape(nd::array("test"), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_string(string_encoding_utf_8)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_string(string_encoding_ascii)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_string(string_encoding_utf_16)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_string(string_encoding_utf_32)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_string(string_encoding_ucs_2)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_fixedstring(1, string_encoding_utf_8)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_fixedstring(10, string_encoding_utf_8)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_fixedstring(10, string_encoding_ascii)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_fixedstring(10, string_encoding_utf_16)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_fixedstring(10, string_encoding_utf_32)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(ndt::make_fixedstring(10, string_encoding_ucs_2)), "", false));
}

TEST(DataShapeFormatter, DTypeStringAtoms) {
    EXPECT_EQ("string", format_datashape(ndt::make_string(), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_string(string_encoding_utf_8), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_string(string_encoding_ascii), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_string(string_encoding_utf_16), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_string(string_encoding_utf_32), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_string(string_encoding_ucs_2), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_fixedstring(1, string_encoding_utf_8), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_fixedstring(10, string_encoding_utf_8), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_fixedstring(10, string_encoding_ascii), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_fixedstring(10, string_encoding_utf_16), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_fixedstring(10, string_encoding_utf_32), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndt::make_fixedstring(10, string_encoding_ucs_2), "", false));
}

TEST(DataShapeFormatter, ArrayUniformArrays) {
  EXPECT_EQ("3 * int32", format_datashape(nd::empty<int32_t[3]>(), "", false));
  EXPECT_EQ(
      "var * int32",
      format_datashape(nd::empty(ndt::make_var_dim(ndt::make_type<int32_t>())),
                       "", false));
  EXPECT_EQ("var * 3 * int32",
            format_datashape(nd::empty(ndt::make_var_dim(ndt::make_cfixed_dim(
                                 3, ndt::make_type<int32_t>()))),
                             "", false));
}

TEST(DataShapeFormatter, DTypeUniformArrays) {
    EXPECT_EQ("strided * strided * strided * int32", format_datashape(
                    ndt::make_strided_dim(ndt::make_type<int32_t>(), 3), "", false));
    EXPECT_EQ("var * int32", format_datashape(
                    ndt::make_var_dim(ndt::make_type<int32_t>()), "", false));
    EXPECT_EQ("var * 3 * int32", format_datashape(
                    ndt::make_var_dim(
                        ndt::make_cfixed_dim(3, ndt::make_type<int32_t>())), "", false));
    EXPECT_EQ("var * 3 * int32", format_datashape(
                    ndt::make_var_dim(
                        ndt::make_fixed_dim(3, ndt::make_type<int32_t>())), "", false));
    EXPECT_EQ("var * strided * int32", format_datashape(
                    ndt::make_var_dim(
                        ndt::make_strided_dim(ndt::make_type<int32_t>())), "", false));
}

TEST(DataShapeFormatter, ArrayStructs) {
    EXPECT_EQ("{x: int32, y: float64}", format_datashape(
                    nd::empty(ndt::make_cstruct(
                                    ndt::make_type<int32_t>(), "x",
                                    ndt::make_type<double>(), "y")), "", false));
    EXPECT_EQ("{x: var * {a: int32, b: int8}, y: 5 * var * uint8}",
                    format_datashape(nd::empty(ndt::make_struct(
                                    ndt::make_var_dim(ndt::make_cstruct(
                                        ndt::make_type<int32_t>(), "a",
                                        ndt::make_type<int8_t>(), "b")), "x",
                                    ndt::make_cfixed_dim(5, ndt::make_var_dim(
                                        ndt::make_type<uint8_t>())), "y")), "", false));
}

TEST(DataShapeFormatter, DTypeStructs) {
    EXPECT_EQ("{x: int32, y: float64}", format_datashape(
                    ndt::make_cstruct(
                                    ndt::make_type<int32_t>(), "x",
                                    ndt::make_type<double>(), "y"), "", false));
    EXPECT_EQ("{x: var * {a: int32, b: int8}, y: 5 * var * uint8}",
                    format_datashape(ndt::make_struct(
                                    ndt::make_var_dim(ndt::make_cstruct(
                                        ndt::make_type<int32_t>(), "a",
                                        ndt::make_type<int8_t>(), "b")), "x",
                                    ndt::make_cfixed_dim(5, ndt::make_var_dim(
                                        ndt::make_type<uint8_t>())), "y"), "", false));
    EXPECT_EQ("{x: strided * {a: int32, b: int8}, y: var * strided * uint8}",
                    format_datashape(ndt::make_struct(
                                    ndt::make_strided_dim(ndt::make_cstruct(
                                        ndt::make_type<int32_t>(), "a",
                                        ndt::make_type<int8_t>(), "b")), "x",
                                    ndt::make_var_dim(ndt::make_strided_dim(
                                        ndt::make_type<uint8_t>())), "y"), "", false));
}
