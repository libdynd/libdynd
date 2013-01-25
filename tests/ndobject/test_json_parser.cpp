//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/json_parser.hpp>
#include <dynd/dtypes/var_array_dtype.hpp>
#include <dynd/dtypes/fixedarray_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(JSONParser, BuiltinsFromBool) {
    ndobject n;

    n = parse_json(make_dtype<dynd_bool>(), "true");
    EXPECT_EQ(make_dtype<dynd_bool>(), n.get_dtype());
    EXPECT_EQ(true, n.as<bool>());
    n = parse_json(make_dtype<dynd_bool>(), "false");
    EXPECT_EQ(make_dtype<dynd_bool>(), n.get_dtype());
    EXPECT_EQ(false, n.as<bool>());
    n = parse_json(make_dtype<dynd_bool>(), "null");
    EXPECT_EQ(make_dtype<dynd_bool>(), n.get_dtype());
    EXPECT_EQ(false, n.as<bool>());

    EXPECT_THROW(parse_json(make_dtype<dynd_bool>(), "flase"), runtime_error);
}

TEST(JSONParser, BuiltinsFromInteger) {
    ndobject n;

    n = parse_json(make_dtype<int8_t>(), "123");
    EXPECT_EQ(make_dtype<int8_t>(), n.get_dtype());
    EXPECT_EQ(123, n.as<int8_t>());
    n = parse_json(make_dtype<int16_t>(), "-30000");
    EXPECT_EQ(make_dtype<int16_t>(), n.get_dtype());
    EXPECT_EQ(-30000, n.as<int16_t>());
    n = parse_json(make_dtype<int32_t>(), "500000");
    EXPECT_EQ(make_dtype<int32_t>(), n.get_dtype());
    EXPECT_EQ(500000, n.as<int32_t>());
    n = parse_json(make_dtype<int64_t>(), "-3000000000");
    EXPECT_EQ(make_dtype<int64_t>(), n.get_dtype());
    EXPECT_EQ(-3000000000LL, n.as<int64_t>());

    n = parse_json(make_dtype<uint8_t>(), "123");
    EXPECT_EQ(make_dtype<uint8_t>(), n.get_dtype());
    EXPECT_EQ(123, n.as<uint8_t>());
    n = parse_json(make_dtype<uint16_t>(), "50000");
    EXPECT_EQ(make_dtype<uint16_t>(), n.get_dtype());
    EXPECT_EQ(50000, n.as<uint16_t>());
    n = parse_json(make_dtype<uint32_t>(), "500000");
    EXPECT_EQ(make_dtype<uint32_t>(), n.get_dtype());
    EXPECT_EQ(500000, n.as<uint32_t>());
    n = parse_json(make_dtype<uint64_t>(), "3000000000");
    EXPECT_EQ(make_dtype<uint64_t>(), n.get_dtype());
    EXPECT_EQ(3000000000LL, n.as<uint64_t>());
}

TEST(JSONParser, BuiltinsFromFloat) {
    ndobject n;

    n = parse_json(make_dtype<float>(), "123");
    EXPECT_EQ(make_dtype<float>(), n.get_dtype());
    EXPECT_EQ(123.f, n.as<float>());
    n = parse_json(make_dtype<float>(), "1.5");
    EXPECT_EQ(make_dtype<float>(), n.get_dtype());
    EXPECT_EQ(1.5f, n.as<float>());
    n = parse_json(make_dtype<float>(), "1.5e2");
    EXPECT_EQ(make_dtype<float>(), n.get_dtype());
    EXPECT_EQ(1.5e2f, n.as<float>());

    n = parse_json(make_dtype<double>(), "123");
    EXPECT_EQ(make_dtype<double>(), n.get_dtype());
    EXPECT_EQ(123., n.as<double>());
    n = parse_json(make_dtype<double>(), "1.5");
    EXPECT_EQ(make_dtype<double>(), n.get_dtype());
    EXPECT_EQ(1.5, n.as<double>());
    n = parse_json(make_dtype<double>(), "1.5e2");
    EXPECT_EQ(make_dtype<double>(), n.get_dtype());
    EXPECT_EQ(1.5e2, n.as<double>());
}

TEST(JSONParser, String) {
    ndobject n;

    n = parse_json(make_string_dtype(string_encoding_utf_8), "\"testing one two three\"");
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8), n.get_dtype());
    EXPECT_EQ("testing one two three", n.as<string>());
    n = parse_json(make_string_dtype(string_encoding_utf_8), "\" \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\u0020 \"");
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8), n.get_dtype());
    EXPECT_EQ(" \" \\ / \b \f \n \r \t   ", n.as<string>());

    n = parse_json(make_string_dtype(string_encoding_utf_16), "\"testing one two three\"");
    EXPECT_EQ(make_string_dtype(string_encoding_utf_16), n.get_dtype());
    EXPECT_EQ("testing one two three", n.as<string>());
    n = parse_json(make_string_dtype(string_encoding_utf_16), "\" \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\u0020 \"");
    EXPECT_EQ(make_string_dtype(string_encoding_utf_16), n.get_dtype());
    EXPECT_EQ(" \" \\ / \b \f \n \r \t   ", n.as<string>());

    EXPECT_THROW(parse_json(make_string_dtype(string_encoding_utf_8), "false"), runtime_error);
}

TEST(JSONParser, ListBools) {
    ndobject n;

    n = parse_json(make_var_array_dtype(make_dtype<dynd_bool>()),
                    "  [true, true, false, null]  ");
    EXPECT_EQ(make_var_array_dtype(make_dtype<dynd_bool>()), n.get_dtype());
    EXPECT_EQ(true, n.at(0).as<bool>());
    EXPECT_EQ(true, n.at(1).as<bool>());
    EXPECT_EQ(false, n.at(2).as<bool>());
    EXPECT_EQ(false, n.at(3).as<bool>());

    n = parse_json(make_fixedarray_dtype(make_dtype<dynd_bool>(),4),
                    "  [true, true, false, null]  ");
    EXPECT_EQ(make_fixedarray_dtype(make_dtype<dynd_bool>(),4), n.get_dtype());
    EXPECT_EQ(true, n.at(0).as<bool>());
    EXPECT_EQ(true, n.at(1).as<bool>());
    EXPECT_EQ(false, n.at(2).as<bool>());
    EXPECT_EQ(false, n.at(3).as<bool>());

    EXPECT_THROW(parse_json(make_var_array_dtype(make_dtype<dynd_bool>()),
                    "[true, true, false, null] 3.5"),
                    runtime_error);
    EXPECT_THROW(parse_json(make_fixedarray_dtype(make_dtype<dynd_bool>(),4),
                    "[true, true, false, null] 3.5"),
                    runtime_error);
    EXPECT_THROW(parse_json(make_fixedarray_dtype(make_dtype<dynd_bool>(),3),
                    "[true, true, false, null]"),
                    runtime_error);
    EXPECT_THROW(parse_json(make_fixedarray_dtype(make_dtype<dynd_bool>(),5),
                    "[true, true, false, null]"),
                    runtime_error);
}

TEST(JSONParser, NestedListInts) {
    ndobject n;

    n = parse_json(make_fixedarray_dtype(make_var_array_dtype(make_dtype<int>()), 3),
                    "  [[1,2,3], [4,5], [6,7,-10,1000] ]  ");
    EXPECT_EQ(make_fixedarray_dtype(make_var_array_dtype(make_dtype<int>()), 3), n.get_dtype());
    EXPECT_EQ(1, n.at(0,0).as<int>());
    EXPECT_EQ(2, n.at(0,1).as<int>());
    EXPECT_EQ(3, n.at(0,2).as<int>());
    EXPECT_EQ(4, n.at(1,0).as<int>());
    EXPECT_EQ(5, n.at(1,1).as<int>());
    EXPECT_EQ(6, n.at(2,0).as<int>());
    EXPECT_EQ(7, n.at(2,1).as<int>());
    EXPECT_EQ(-10, n.at(2,2).as<int>());
    EXPECT_EQ(1000, n.at(2,3).as<int>());
}