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
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/json_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(JSONParser, BuiltinsFromBool) {
    ndobject n;

    n = parse_json(make_dtype<dynd_bool>(), "true");
    EXPECT_EQ(make_dtype<dynd_bool>(), n.get_dtype());
    EXPECT_TRUE(n.as<bool>());
    n = parse_json(make_dtype<dynd_bool>(), "false");
    EXPECT_EQ(make_dtype<dynd_bool>(), n.get_dtype());
    EXPECT_FALSE(n.as<bool>());
    n = parse_json(make_dtype<dynd_bool>(), "null");
    EXPECT_EQ(make_dtype<dynd_bool>(), n.get_dtype());
    EXPECT_FALSE(n.as<bool>());

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
    EXPECT_EQ(123u, n.as<uint8_t>());
    n = parse_json(make_dtype<uint16_t>(), "50000");
    EXPECT_EQ(make_dtype<uint16_t>(), n.get_dtype());
    EXPECT_EQ(50000u, n.as<uint16_t>());
    n = parse_json(make_dtype<uint32_t>(), "500000");
    EXPECT_EQ(make_dtype<uint32_t>(), n.get_dtype());
    EXPECT_EQ(500000u, n.as<uint32_t>());
    n = parse_json(make_dtype<uint64_t>(), "3000000000");
    EXPECT_EQ(make_dtype<uint64_t>(), n.get_dtype());
    EXPECT_EQ(3000000000ULL, n.as<uint64_t>());
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
    EXPECT_TRUE(n.at(0).as<bool>());
    EXPECT_TRUE(n.at(1).as<bool>());
    EXPECT_FALSE(n.at(2).as<bool>());
    EXPECT_FALSE(n.at(3).as<bool>());

    n = parse_json(make_fixedarray_dtype(4, make_dtype<dynd_bool>()),
                    "  [true, true, false, null]  ");
    EXPECT_EQ(make_fixedarray_dtype(4, make_dtype<dynd_bool>()), n.get_dtype());
    EXPECT_TRUE(n.at(0).as<bool>());
    EXPECT_TRUE(n.at(1).as<bool>());
    EXPECT_FALSE(n.at(2).as<bool>());
    EXPECT_FALSE(n.at(3).as<bool>());

    EXPECT_THROW(parse_json(make_var_array_dtype(make_dtype<dynd_bool>()),
                    "[true, true, false, null] 3.5"),
                    runtime_error);
    EXPECT_THROW(parse_json(make_fixedarray_dtype(4, make_dtype<dynd_bool>()),
                    "[true, true, false, null] 3.5"),
                    runtime_error);
    EXPECT_THROW(parse_json(make_fixedarray_dtype(3, make_dtype<dynd_bool>()),
                    "[true, true, false, null]"),
                    runtime_error);
    EXPECT_THROW(parse_json(make_fixedarray_dtype(5, make_dtype<dynd_bool>()),
                    "[true, true, false, null]"),
                    runtime_error);
}

TEST(JSONParser, NestedListInts) {
    ndobject n;

    n = parse_json(make_fixedarray_dtype(3, make_var_array_dtype(make_dtype<int>())),
                    "  [[1,2,3], [4,5], [6,7,-10,1000] ]  ");
    EXPECT_EQ(make_fixedarray_dtype(3, make_var_array_dtype(make_dtype<int>())), n.get_dtype());
    EXPECT_EQ(1, n.at(0,0).as<int>());
    EXPECT_EQ(2, n.at(0,1).as<int>());
    EXPECT_EQ(3, n.at(0,2).as<int>());
    EXPECT_EQ(4, n.at(1,0).as<int>());
    EXPECT_EQ(5, n.at(1,1).as<int>());
    EXPECT_EQ(6, n.at(2,0).as<int>());
    EXPECT_EQ(7, n.at(2,1).as<int>());
    EXPECT_EQ(-10, n.at(2,2).as<int>());
    EXPECT_EQ(1000, n.at(2,3).as<int>());

    n = parse_json(make_var_array_dtype(make_fixedarray_dtype(3, make_dtype<int>())),
                    "  [[1,2,3], [4,5,2] ]  ");
    EXPECT_EQ(make_var_array_dtype(make_fixedarray_dtype(3, make_dtype<int>())), n.get_dtype());
    EXPECT_EQ(1, n.at(0,0).as<int>());
    EXPECT_EQ(2, n.at(0,1).as<int>());
    EXPECT_EQ(3, n.at(0,2).as<int>());
    EXPECT_EQ(4, n.at(1,0).as<int>());
    EXPECT_EQ(5, n.at(1,1).as<int>());
    EXPECT_EQ(2, n.at(1,2).as<int>());
}

TEST(JSONParser, Struct) {
    ndobject n;
    dtype sdt = make_fixedstruct_dtype(make_dtype<int>(), "id", make_dtype<double>(), "amount",
                    make_string_dtype(), "name", make_date_dtype(), "when");

    // A straightforward struct
    n = parse_json(sdt, "{\"amount\":3.75,\"id\":24601,"
                    " \"when\":\"2012-09-19\",\"name\":\"Jean\"}");
    EXPECT_EQ(sdt, n.get_dtype());
    EXPECT_EQ(24601,        n.at(0).as<int>());
    EXPECT_EQ(3.75,         n.at(1).as<double>());
    EXPECT_EQ("Jean",       n.at(2).as<string>());
    EXPECT_EQ("2012-09-19", n.at(3).as<string>());

    // Default parsing policy discards extra JSON fields
    n = parse_json(sdt, "{\"amount\":3.75,\"id\":24601,\"discarded\":[1,2,3],"
                    " \"when\":\"2012-09-19\",\"name\":\"Jean\"}");
    EXPECT_EQ(sdt, n.get_dtype());
    EXPECT_EQ(24601,        n.at(0).as<int>());
    EXPECT_EQ(3.75,         n.at(1).as<double>());
    EXPECT_EQ("Jean",       n.at(2).as<string>());
    EXPECT_EQ("2012-09-19", n.at(3).as<string>());

    // Every field must be populated, though
    EXPECT_THROW(parse_json(sdt, "{\"amount\":3.75,\"discarded\":[1,2,3],"
                    " \"when\":\"2012-09-19\",\"name\":\"Jean\"}"),
                    runtime_error);
}

TEST(JSONParser, NestedStruct) {
    ndobject n;
    dtype sdt = make_fixedstruct_dtype(make_fixedarray_dtype(3, make_dtype<float>()), "position",
                    make_dtype<double>(), "amount",
                    make_fixedstruct_dtype(make_string_dtype(), "name", make_date_dtype(), "when"), "data");

    n = parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, "
                    "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] }");
    EXPECT_EQ(sdt, n.get_dtype());
    EXPECT_EQ(3.5,          n.at(0,0).as<float>());
    EXPECT_EQ(1.0,          n.at(0,1).as<float>());
    EXPECT_EQ(1e10,         n.at(0,2).as<float>());
    EXPECT_EQ(10.5,         n.at(1).as<double>());
    EXPECT_EQ("Harvey",     n.at(2,0).as<string>());
    EXPECT_EQ("1970-02-13", n.at(2,1).as<string>());

    // Too many entries in "position"
    EXPECT_THROW(parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, "
                    "\"amount\": 10.5, \"position\": [1.5,3.5,1.0,1e10] }"),
                    runtime_error);

    // Too few entries in "position"
    EXPECT_THROW(parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, "
                    "\"amount\": 10.5, \"position\": [1.0,1e10] }"),
                    runtime_error);

    // Missing field "when"
    EXPECT_THROW(parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when2\":\"1970-02-13\"}, "
                    "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] }"),
                    runtime_error);
}

TEST(JSONParser, ListOfStruct) {
    ndobject n;
    dtype sdt = make_var_array_dtype(make_fixedstruct_dtype(make_fixedarray_dtype(3, make_dtype<float>()), "position",
                    make_dtype<double>(), "amount",
                    make_fixedstruct_dtype(make_string_dtype(), "name", make_date_dtype(), "when"), "data"));

    n = parse_json(sdt, "[{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, \n"
                    "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] },\n"
                    "{\"position\":[1,2,3], \"amount\": 3.125,\n"
                    "\"data\":{ \"when\":\"2013-12-25\", \"name\":\"Frank\"}}]");
    EXPECT_EQ(3.5,          n.at(0,0,0).as<float>());
    EXPECT_EQ(1.0,          n.at(0,0,1).as<float>());
    EXPECT_EQ(1e10,         n.at(0,0,2).as<float>());
    EXPECT_EQ(10.5,         n.at(0,1).as<double>());
    EXPECT_EQ("Harvey",     n.at(0,2,0).as<string>());
    EXPECT_EQ("1970-02-13", n.at(0,2,1).as<string>());
    EXPECT_EQ(1,            n.at(1,0,0).as<float>());
    EXPECT_EQ(2,            n.at(1,0,1).as<float>());
    EXPECT_EQ(3,            n.at(1,0,2).as<float>());
    EXPECT_EQ(3.125,        n.at(1,1).as<double>());
    EXPECT_EQ("Frank",      n.at(1,2,0).as<string>());
    EXPECT_EQ("2013-12-25", n.at(1,2,1).as<string>());

    // Spurious '#' inserted
    EXPECT_THROW(parse_json(sdt, "[{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, \n"
                    "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] },\n"
                    "{\"position\":[1,2,3], \"amount\": 3.125#,\n"
                    "\"data\":{ \"when\":\"2013-12-25\", \"name\":\"Frank\"}}]"),
                    runtime_error);
}

TEST(JSONParser, JSONDType) {
    ndobject n;

    // Parsing JSON with the output being just a json string
    n = parse_json("json", "{\"a\":3.14}");
    EXPECT_EQ(make_json_dtype(), n.get_dtype());
    EXPECT_EQ("{\"a\":3.14}", n.as<string>());

    // Parsing JSON with a piece of it being a json string
    n = parse_json("{a: json; b: int32; c: string}",
                    "{\"c\": \"testing string\", \"a\": [3.1, {\"X\":2}, [1,2]], \"b\":12}");
    EXPECT_EQ(make_fixedstruct_dtype(make_json_dtype(), "a", make_dtype<int32_t>(), "b", make_string_dtype(), "c"),
                    n.get_dtype());
    EXPECT_EQ("[3.1, {\"X\":2}, [1,2]]", n.at(0).as<string>());
    EXPECT_EQ(12, n.at(1).as<int>());
    EXPECT_EQ("testing string", n.at(2).as<string>());
}
