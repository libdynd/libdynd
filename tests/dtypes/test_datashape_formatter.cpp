//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtypes/datashape_formatter.hpp>
#include <dynd/dtypes/var_array_dtype.hpp>
#include <dynd/dtypes/fixedarray_dtype.hpp>
#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(DataShapeFormatter, BuiltinAtoms) {
    EXPECT_EQ("bool", format_datashape(ndobject(true), "", false));
    EXPECT_EQ("int8", format_datashape(ndobject((int8_t)0), "", false));
    EXPECT_EQ("int16", format_datashape(ndobject((int16_t)0), "", false));
    EXPECT_EQ("int32", format_datashape(ndobject((int32_t)0), "", false));
    EXPECT_EQ("int64", format_datashape(ndobject((int64_t)0), "", false));
    EXPECT_EQ("uint8", format_datashape(ndobject((uint8_t)0), "", false));
    EXPECT_EQ("uint16", format_datashape(ndobject((uint16_t)0), "", false));
    EXPECT_EQ("uint32", format_datashape(ndobject((uint32_t)0), "", false));
    EXPECT_EQ("uint64", format_datashape(ndobject((uint64_t)0), "", false));
    EXPECT_EQ("float32", format_datashape(ndobject(0.f), "", false));
    EXPECT_EQ("float64", format_datashape(ndobject(0.), "", false));
    EXPECT_EQ("cfloat32", format_datashape(ndobject(complex<float>(0.f)), "", false));
    EXPECT_EQ("cfloat64", format_datashape(ndobject(complex<double>(0.)), "", false));
}

TEST(DataShapeFormatter, StringAtoms) {
    EXPECT_EQ("string", format_datashape(ndobject("test"), "", false));
    EXPECT_EQ("string", format_datashape(
                    ndobject(make_string_dtype(string_encoding_utf_8)), "", false));
    EXPECT_EQ("string('A')", format_datashape(
                    ndobject(make_string_dtype(string_encoding_ascii)), "", false));
    EXPECT_EQ("string('U16')", format_datashape(
                    ndobject(make_string_dtype(string_encoding_utf_16)), "", false));
    EXPECT_EQ("string('U32')", format_datashape(
                    ndobject(make_string_dtype(string_encoding_utf_32)), "", false));
    EXPECT_EQ("string('ucs2')", format_datashape(
                    ndobject(make_string_dtype(string_encoding_ucs_2)), "", false));
    EXPECT_EQ("string(1)", format_datashape(
                    ndobject(make_fixedstring_dtype(1, string_encoding_utf_8)), "", false));
    EXPECT_EQ("string(10)", format_datashape(
                    ndobject(make_fixedstring_dtype(10, string_encoding_utf_8)), "", false));
    EXPECT_EQ("string(10,'A')", format_datashape(
                    ndobject(make_fixedstring_dtype(10, string_encoding_ascii)), "", false));
    EXPECT_EQ("string(10,'U16')", format_datashape(
                    ndobject(make_fixedstring_dtype(10, string_encoding_utf_16)), "", false));
    EXPECT_EQ("string(10,'U32')", format_datashape(
                    ndobject(make_fixedstring_dtype(10, string_encoding_utf_32)), "", false));
    EXPECT_EQ("string(10,'ucs2')", format_datashape(
                    ndobject(make_fixedstring_dtype(10, string_encoding_ucs_2)), "", false));
}

TEST(DataShapeFormatter, UniformArrays) {
    EXPECT_EQ("3, int32", format_datashape(
                    make_strided_ndobject(3, make_dtype<int32_t>()), "", false));
    EXPECT_EQ("VarDim, int32", format_datashape(
                    ndobject(make_var_array_dtype(make_dtype<int32_t>())), "", false));
    EXPECT_EQ("VarDim, 3, int32", format_datashape(
                    ndobject(make_var_array_dtype(
                        make_fixedarray_dtype(3, make_dtype<int32_t>()))), "", false));
}

TEST(DataShapeFormatter, Structs) {
    EXPECT_EQ("{x: int32; y: float64}", format_datashape(
                    ndobject(make_fixedstruct_dtype(
                                    make_dtype<int32_t>(), "x",
                                    make_dtype<double>(), "y")), "", false));
    EXPECT_EQ("{x: VarDim, {a: int32; b: int8}; y: 5, VarDim, uint8}",
                    format_datashape(ndobject(make_struct_dtype(
                                    make_var_array_dtype(make_fixedstruct_dtype(
                                        make_dtype<int32_t>(), "a",
                                        make_dtype<int8_t>(), "b")), "x",
                                    make_fixedarray_dtype(5, make_var_array_dtype(
                                        make_dtype<uint8_t>())), "y")), "", false));
}