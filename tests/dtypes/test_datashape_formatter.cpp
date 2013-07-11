//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtypes/datashape_formatter.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/struct_type.hpp>
#include <dynd/dtypes/cstruct_type.hpp>
#include <dynd/dtypes/string_type.hpp>
#include <dynd/dtypes/fixedstring_type.hpp>

using namespace std;
using namespace dynd;

TEST(DataShapeFormatter, ArrayBuiltinAtoms) {
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
    EXPECT_EQ("float16", format_datashape(nd::array(dynd_float16(0.f, assign_error_none)), "", false));
    EXPECT_EQ("float32", format_datashape(nd::array(0.f), "", false));
    EXPECT_EQ("float64", format_datashape(nd::array(0.), "", false));
    EXPECT_EQ("cfloat32", format_datashape(nd::array(complex<float>(0.f)), "", false));
    EXPECT_EQ("cfloat64", format_datashape(nd::array(complex<double>(0.)), "", false));
}

TEST(DataShapeFormatter, DTypeBuiltinAtoms) {
    EXPECT_EQ("bool", format_datashape(ndt::make_dtype<dynd_bool>(), "", false));
    EXPECT_EQ("int8", format_datashape(ndt::make_dtype<int8_t>(), "", false));
    EXPECT_EQ("int16", format_datashape(ndt::make_dtype<int16_t>(), "", false));
    EXPECT_EQ("int32", format_datashape(ndt::make_dtype<int32_t>(), "", false));
    EXPECT_EQ("int64", format_datashape(ndt::make_dtype<int64_t>(), "", false));
    EXPECT_EQ("uint8", format_datashape(ndt::make_dtype<uint8_t>(), "", false));
    EXPECT_EQ("uint16", format_datashape(ndt::make_dtype<uint16_t>(), "", false));
    EXPECT_EQ("uint32", format_datashape(ndt::make_dtype<uint32_t>(), "", false));
    EXPECT_EQ("uint64", format_datashape(ndt::make_dtype<uint64_t>(), "", false));
    EXPECT_EQ("float32", format_datashape(ndt::make_dtype<float>(), "", false));
    EXPECT_EQ("float64", format_datashape(ndt::make_dtype<double>(), "", false));
    EXPECT_EQ("cfloat32", format_datashape(ndt::make_dtype<complex<float> >(), "", false));
    EXPECT_EQ("cfloat64", format_datashape(ndt::make_dtype<complex<double> >(), "", false));
}

TEST(DataShapeFormatter, ArrayStringAtoms) {
    EXPECT_EQ("string", format_datashape(nd::array("test"), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_string_type(string_encoding_utf_8)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_string_type(string_encoding_ascii)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_string_type(string_encoding_utf_16)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_string_type(string_encoding_utf_32)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_string_type(string_encoding_ucs_2)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_fixedstring_type(1, string_encoding_utf_8)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_fixedstring_type(10, string_encoding_utf_8)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_fixedstring_type(10, string_encoding_ascii)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_fixedstring_type(10, string_encoding_utf_16)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_fixedstring_type(10, string_encoding_utf_32)), "", false));
    EXPECT_EQ("string", format_datashape(
                    nd::empty(make_fixedstring_type(10, string_encoding_ucs_2)), "", false));
}

TEST(DataShapeFormatter, DTypeStringAtoms) {
    EXPECT_EQ("string", format_datashape(make_string_type(), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_string_type(string_encoding_utf_8), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_string_type(string_encoding_ascii), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_string_type(string_encoding_utf_16), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_string_type(string_encoding_utf_32), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_string_type(string_encoding_ucs_2), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_fixedstring_type(1, string_encoding_utf_8), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_fixedstring_type(10, string_encoding_utf_8), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_fixedstring_type(10, string_encoding_ascii), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_fixedstring_type(10, string_encoding_utf_16), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_fixedstring_type(10, string_encoding_utf_32), "", false));
    EXPECT_EQ("string", format_datashape(
                    make_fixedstring_type(10, string_encoding_ucs_2), "", false));
}

TEST(DataShapeFormatter, ArrayUniformArrays) {
    EXPECT_EQ("3, int32", format_datashape(
                    nd::make_strided_array(3, ndt::make_dtype<int32_t>()), "", false));
    EXPECT_EQ("var, int32", format_datashape(
                    nd::empty(make_var_dim_dtype(ndt::make_dtype<int32_t>())), "", false));
    EXPECT_EQ("var, 3, int32", format_datashape(
                    nd::empty(make_var_dim_dtype(
                        make_fixed_dim_dtype(3, ndt::make_dtype<int32_t>()))), "", false));
}

TEST(DataShapeFormatter, DTypeUniformArrays) {
    EXPECT_EQ("A, B, C, int32", format_datashape(
                    make_strided_dim_dtype(ndt::make_dtype<int32_t>(), 3), "", false));
    EXPECT_EQ("var, int32", format_datashape(
                    make_var_dim_dtype(ndt::make_dtype<int32_t>()), "", false));
    EXPECT_EQ("var, 3, int32", format_datashape(
                    make_var_dim_dtype(
                        make_fixed_dim_dtype(3, ndt::make_dtype<int32_t>())), "", false));
    EXPECT_EQ("var, A, int32", format_datashape(
                    make_var_dim_dtype(
                        make_strided_dim_dtype(ndt::make_dtype<int32_t>())), "", false));
}

TEST(DataShapeFormatter, ArrayStructs) {
    EXPECT_EQ("{x: int32; y: float64}", format_datashape(
                    nd::empty(make_cstruct_type(
                                    ndt::make_dtype<int32_t>(), "x",
                                    ndt::make_dtype<double>(), "y")), "", false));
    EXPECT_EQ("{x: var, {a: int32; b: int8}; y: 5, var, uint8}",
                    format_datashape(nd::empty(make_struct_type(
                                    make_var_dim_dtype(make_cstruct_type(
                                        ndt::make_dtype<int32_t>(), "a",
                                        ndt::make_dtype<int8_t>(), "b")), "x",
                                    make_fixed_dim_dtype(5, make_var_dim_dtype(
                                        ndt::make_dtype<uint8_t>())), "y")), "", false));
}

TEST(DataShapeFormatter, DTypeStructs) {
    EXPECT_EQ("{x: int32; y: float64}", format_datashape(
                    make_cstruct_type(
                                    ndt::make_dtype<int32_t>(), "x",
                                    ndt::make_dtype<double>(), "y"), "", false));
    EXPECT_EQ("{x: var, {a: int32; b: int8}; y: 5, var, uint8}",
                    format_datashape(make_struct_type(
                                    make_var_dim_dtype(make_cstruct_type(
                                        ndt::make_dtype<int32_t>(), "a",
                                        ndt::make_dtype<int8_t>(), "b")), "x",
                                    make_fixed_dim_dtype(5, make_var_dim_dtype(
                                        ndt::make_dtype<uint8_t>())), "y"), "", false));
    EXPECT_EQ("{x: A, {a: int32; b: int8}; y: var, B, uint8}",
                    format_datashape(make_struct_type(
                                    make_strided_dim_dtype(make_cstruct_type(
                                        ndt::make_dtype<int32_t>(), "a",
                                        ndt::make_dtype<int8_t>(), "b")), "x",
                                    make_var_dim_dtype(make_strided_dim_dtype(
                                        ndt::make_dtype<uint8_t>())), "y"), "", false));
}
