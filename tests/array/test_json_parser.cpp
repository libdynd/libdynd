//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/json_parser.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/json_type.hpp>

using namespace std;
using namespace dynd;

TEST(JSONParser, BuiltinsFromBool) {
    nd::array n;

    n = parse_json(ndt::make_type<dynd_bool>(), "true");
    EXPECT_EQ(ndt::make_type<dynd_bool>(), n.get_type());
    EXPECT_TRUE(n.as<bool>());
    n = parse_json(ndt::make_type<dynd_bool>(), "false");
    EXPECT_EQ(ndt::make_type<dynd_bool>(), n.get_type());
    EXPECT_FALSE(n.as<bool>());
    n = parse_json(ndt::make_type<dynd_bool>(), "null");
    EXPECT_EQ(ndt::make_type<dynd_bool>(), n.get_type());
    EXPECT_FALSE(n.as<bool>());

    EXPECT_THROW(parse_json(ndt::make_type<dynd_bool>(), "flase"), invalid_argument);
}

TEST(JSONParser, BuiltinsFromInteger) {
    nd::array n;

    n = parse_json(ndt::make_type<int8_t>(), "123");
    EXPECT_EQ(ndt::make_type<int8_t>(), n.get_type());
    EXPECT_EQ(123, n.as<int8_t>());
    n = parse_json(ndt::make_type<int16_t>(), "-30000");
    EXPECT_EQ(ndt::make_type<int16_t>(), n.get_type());
    EXPECT_EQ(-30000, n.as<int16_t>());
    n = parse_json(ndt::make_type<int32_t>(), "500000");
    EXPECT_EQ(ndt::make_type<int32_t>(), n.get_type());
    EXPECT_EQ(500000, n.as<int32_t>());
    n = parse_json(ndt::make_type<int64_t>(), "-3000000000");
    EXPECT_EQ(ndt::make_type<int64_t>(), n.get_type());
    EXPECT_EQ(-3000000000LL, n.as<int64_t>());

    n = parse_json(ndt::make_type<uint8_t>(), "123");
    EXPECT_EQ(ndt::make_type<uint8_t>(), n.get_type());
    EXPECT_EQ(123u, n.as<uint8_t>());
    n = parse_json(ndt::make_type<uint16_t>(), "50000");
    EXPECT_EQ(ndt::make_type<uint16_t>(), n.get_type());
    EXPECT_EQ(50000u, n.as<uint16_t>());
    n = parse_json(ndt::make_type<uint32_t>(), "500000");
    EXPECT_EQ(ndt::make_type<uint32_t>(), n.get_type());
    EXPECT_EQ(500000u, n.as<uint32_t>());
    n = parse_json(ndt::make_type<uint64_t>(), "3000000000");
    EXPECT_EQ(ndt::make_type<uint64_t>(), n.get_type());
    EXPECT_EQ(3000000000ULL, n.as<uint64_t>());
}

TEST(JSONParser, SignedIntegerLimits) {
    nd::array n;

    n = parse_json(ndt::make_type<int8_t>(), "-128");
    EXPECT_EQ(-128, n.as<int8_t>());
    n = parse_json(ndt::make_type<int8_t>(), "127");
    EXPECT_EQ(127, n.as<int8_t>());
    n = parse_json(ndt::make_type<int16_t>(), "-32768");
    EXPECT_EQ(-32768, n.as<int16_t>());
    n = parse_json(ndt::make_type<int16_t>(), "32767");
    EXPECT_EQ(32767, n.as<int16_t>());
    n = parse_json(ndt::make_type<int32_t>(), "-2147483648");
    EXPECT_EQ(-2147483648LL, n.as<int32_t>());
    n = parse_json(ndt::make_type<int32_t>(), "2147483647");
    EXPECT_EQ(2147483647, n.as<int32_t>());
    n = parse_json(ndt::make_type<int64_t>(), "-9223372036854775808");
    EXPECT_EQ(-9223372036854775807LL - 1, n.as<int64_t>());
    n = parse_json(ndt::make_type<int64_t>(), "9223372036854775807");
    EXPECT_EQ(9223372036854775807LL, n.as<int64_t>());
    EXPECT_THROW(parse_json(ndt::make_type<int8_t>(), "-129"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<int8_t>(), "128"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<int16_t>(), "-32769"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<int16_t>(), "32768"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<int32_t>(), "-2147483649"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<int32_t>(), "2147483648"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<int64_t>(), "-9223372036854775809"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<int64_t>(), "9223372036854775808"), exception);
}

TEST(JSONParser, UnsignedIntegerLimits) {
    nd::array n;

    n = parse_json(ndt::make_type<uint8_t>(), "0");
    EXPECT_EQ(0u, n.as<uint8_t>());
    n = parse_json(ndt::make_type<uint8_t>(), "-0");
    EXPECT_EQ(0u, n.as<uint8_t>());
    n = parse_json(ndt::make_type<uint8_t>(), "255");
    EXPECT_EQ(255, n.as<uint8_t>());
    n = parse_json(ndt::make_type<uint16_t>(), "0");
    EXPECT_EQ(0u, n.as<uint16_t>());
    n = parse_json(ndt::make_type<uint16_t>(), "-0");
    EXPECT_EQ(0u, n.as<uint16_t>());
    n = parse_json(ndt::make_type<uint16_t>(), "65535");
    EXPECT_EQ(65535, n.as<uint16_t>());
    n = parse_json(ndt::make_type<uint32_t>(), "0");
    EXPECT_EQ(0u, n.as<uint32_t>());
    n = parse_json(ndt::make_type<uint32_t>(), "-0");
    EXPECT_EQ(0u, n.as<uint32_t>());
    n = parse_json(ndt::make_type<uint32_t>(), "4294967295");
    EXPECT_EQ(4294967295, n.as<uint32_t>());
    n = parse_json(ndt::make_type<uint64_t>(), "0");
    EXPECT_EQ(0u, n.as<uint64_t>());
    n = parse_json(ndt::make_type<uint64_t>(), "-0");
    EXPECT_EQ(0u, n.as<uint64_t>());
    n = parse_json(ndt::make_type<uint64_t>(), "18446744073709551615");
    EXPECT_EQ(18446744073709551615ULL, n.as<uint64_t>());
    EXPECT_THROW(parse_json(ndt::make_type<uint8_t>(), "-1"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<uint8_t>(), "256"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<uint16_t>(), "-1"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<uint16_t>(), "65536"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<uint32_t>(), "-1"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<uint32_t>(), "4294967296"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<uint64_t>(), "-1"), exception);
    EXPECT_THROW(parse_json(ndt::make_type<uint64_t>(), "18446744073709551616"), exception);
}

TEST(JSONParser, BuiltinsFromFloat) {
    nd::array n;

    n = parse_json(ndt::make_type<float>(), "123");
    EXPECT_EQ(ndt::make_type<float>(), n.get_type());
    EXPECT_EQ(123.f, n.as<float>());
    n = parse_json(ndt::make_type<float>(), "1.5");
    EXPECT_EQ(ndt::make_type<float>(), n.get_type());
    EXPECT_EQ(1.5f, n.as<float>());
    n = parse_json(ndt::make_type<float>(), "1.5e2");
    EXPECT_EQ(ndt::make_type<float>(), n.get_type());
    EXPECT_EQ(1.5e2f, n.as<float>());

    n = parse_json(ndt::make_type<double>(), "123");
    EXPECT_EQ(ndt::make_type<double>(), n.get_type());
    EXPECT_EQ(123., n.as<double>());
    n = parse_json(ndt::make_type<double>(), "1.5");
    EXPECT_EQ(ndt::make_type<double>(), n.get_type());
    EXPECT_EQ(1.5, n.as<double>());
    n = parse_json(ndt::make_type<double>(), "1.5e2");
    EXPECT_EQ(ndt::make_type<double>(), n.get_type());
    EXPECT_EQ(1.5e2, n.as<double>());
}

TEST(JSONParser, String) {
    nd::array n;

    n = parse_json(ndt::make_string(string_encoding_utf_8), "\"testing one two three\"");
    EXPECT_EQ(ndt::make_string(string_encoding_utf_8), n.get_type());
    EXPECT_EQ("testing one two three", n.as<string>());
    n = parse_json(ndt::make_string(string_encoding_utf_8), "\" \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\u0020 \"");
    EXPECT_EQ(ndt::make_string(string_encoding_utf_8), n.get_type());
    EXPECT_EQ(" \" \\ / \b \f \n \r \t   ", n.as<string>());

    n = parse_json(ndt::make_string(string_encoding_utf_16), "\"testing one two three\"");
    EXPECT_EQ(ndt::make_string(string_encoding_utf_16), n.get_type());
    EXPECT_EQ("testing one two three", n.as<string>());
    n = parse_json(ndt::make_string(string_encoding_utf_16), "\" \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\u0020 \"");
    EXPECT_EQ(ndt::make_string(string_encoding_utf_16), n.get_type());
    EXPECT_EQ(" \" \\ / \b \f \n \r \t   ", n.as<string>());

    EXPECT_THROW(parse_json(ndt::make_string(string_encoding_utf_8), "false"), invalid_argument);
}

TEST(JSONParser, ListBools) {
    nd::array n;

    n = parse_json(ndt::make_var_dim(ndt::make_type<dynd_bool>()),
                    "  [true, true, false, null]  ");
    EXPECT_EQ(ndt::make_var_dim(ndt::make_type<dynd_bool>()), n.get_type());
    EXPECT_TRUE(n(0).as<bool>());
    EXPECT_TRUE(n(1).as<bool>());
    EXPECT_FALSE(n(2).as<bool>());
    EXPECT_FALSE(n(3).as<bool>());

    n = parse_json(ndt::make_cfixed_dim(4, ndt::make_type<dynd_bool>()),
                    "  [true, true, false, null]  ");
    EXPECT_EQ(ndt::make_cfixed_dim(4, ndt::make_type<dynd_bool>()), n.get_type());
    EXPECT_TRUE(n(0).as<bool>());
    EXPECT_TRUE(n(1).as<bool>());
    EXPECT_FALSE(n(2).as<bool>());
    EXPECT_FALSE(n(3).as<bool>());

    EXPECT_THROW(parse_json(ndt::make_var_dim(ndt::make_type<dynd_bool>()),
                    "[true, true, false, null] 3.5"),
                    invalid_argument);
    EXPECT_THROW(parse_json(ndt::make_cfixed_dim(4, ndt::make_type<dynd_bool>()),
                    "[true, true, false, null] 3.5"),
                    invalid_argument);
    EXPECT_THROW(parse_json(ndt::make_cfixed_dim(3, ndt::make_type<dynd_bool>()),
                    "[true, true, false, null]"),
                    invalid_argument);
    EXPECT_THROW(parse_json(ndt::make_cfixed_dim(5, ndt::make_type<dynd_bool>()),
                    "[true, true, false, null]"),
                    invalid_argument);
}

TEST(JSONParser, NestedListInts) {
    nd::array n;

    n = parse_json(ndt::make_cfixed_dim(3, ndt::make_var_dim(ndt::make_type<int>())),
                    "  [[1,2,3], [4,5], [6,7,-10,1000] ]  ");
    EXPECT_EQ(ndt::make_cfixed_dim(3, ndt::make_var_dim(ndt::make_type<int>())), n.get_type());
    EXPECT_EQ(1, n(0,0).as<int>());
    EXPECT_EQ(2, n(0,1).as<int>());
    EXPECT_EQ(3, n(0,2).as<int>());
    EXPECT_EQ(4, n(1,0).as<int>());
    EXPECT_EQ(5, n(1,1).as<int>());
    EXPECT_EQ(6, n(2,0).as<int>());
    EXPECT_EQ(7, n(2,1).as<int>());
    EXPECT_EQ(-10, n(2,2).as<int>());
    EXPECT_EQ(1000, n(2,3).as<int>());

    n = parse_json(ndt::make_var_dim(ndt::make_cfixed_dim(3, ndt::make_type<int>())),
                    "  [[1,2,3], [4,5,2] ]  ");
    EXPECT_EQ(ndt::make_var_dim(ndt::make_cfixed_dim(3, ndt::make_type<int>())), n.get_type());
    EXPECT_EQ(1, n(0,0).as<int>());
    EXPECT_EQ(2, n(0,1).as<int>());
    EXPECT_EQ(3, n(0,2).as<int>());
    EXPECT_EQ(4, n(1,0).as<int>());
    EXPECT_EQ(5, n(1,1).as<int>());
    EXPECT_EQ(2, n(1,2).as<int>());
}

TEST(JSONParser, Struct) {
    nd::array n;
    ndt::type sdt = ndt::make_cstruct(ndt::make_type<int>(), "id", ndt::make_type<double>(), "amount",
                    ndt::make_string(), "name", ndt::make_date(), "when");

    // A straightforward struct
    n = parse_json(sdt, "{\"amount\":3.75,\"id\":24601,"
                    " \"when\":\"2012-09-19\",\"name\":\"Jean\"}");
    EXPECT_EQ(sdt, n.get_type());
    EXPECT_EQ(24601,        n(0).as<int>());
    EXPECT_EQ(3.75,         n(1).as<double>());
    EXPECT_EQ("Jean",       n(2).as<string>());
    EXPECT_EQ("2012-09-19", n(3).as<string>());

    // Default parsing policy discards extra JSON fields
    n = parse_json(sdt, "{\"amount\":3.75,\"id\":24601,\"discarded\":[1,2,3],"
                    " \"when\":\"2012-09-19\",\"name\":\"Jean\"}");
    EXPECT_EQ(sdt, n.get_type());
    EXPECT_EQ(24601,        n(0).as<int>());
    EXPECT_EQ(3.75,         n(1).as<double>());
    EXPECT_EQ("Jean",       n(2).as<string>());
    EXPECT_EQ("2012-09-19", n(3).as<string>());

    // Every field must be populated, though
    EXPECT_THROW(parse_json(sdt, "{\"amount\":3.75,\"discarded\":[1,2,3],"
                    " \"when\":\"2012-09-19\",\"name\":\"Jean\"}"),
                    invalid_argument);
}

TEST(JSONParser, NestedStruct) {
    nd::array n;
    ndt::type sdt = ndt::make_cstruct(ndt::make_cfixed_dim(3, ndt::make_type<float>()), "position",
                    ndt::make_type<double>(), "amount",
                    ndt::make_cstruct(ndt::make_string(), "name", ndt::make_date(), "when"), "data");

    n = parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, "
                    "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] }");
    EXPECT_EQ(sdt, n.get_type());
    EXPECT_EQ(3.5,          n(0,0).as<float>());
    EXPECT_EQ(1.0,          n(0,1).as<float>());
    EXPECT_EQ(1e10,         n(0,2).as<float>());
    EXPECT_EQ(10.5,         n(1).as<double>());
    EXPECT_EQ("Harvey",     n(2,0).as<string>());
    EXPECT_EQ("1970-02-13", n(2,1).as<string>());

    // Too many entries in "position"
    EXPECT_THROW(parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, "
                    "\"amount\": 10.5, \"position\": [1.5,3.5,1.0,1e10] }"),
                    invalid_argument);

    // Too few entries in "position"
    EXPECT_THROW(parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, "
                    "\"amount\": 10.5, \"position\": [1.0,1e10] }"),
                    invalid_argument);

    // Missing field "when"
    EXPECT_THROW(parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when2\":\"1970-02-13\"}, "
                    "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] }"),
                    invalid_argument);
}

TEST(JSONParser, ListOfStruct) {
    nd::array n;
    ndt::type sdt = ndt::make_var_dim(ndt::make_cstruct(ndt::make_cfixed_dim(3, ndt::make_type<float>()), "position",
                    ndt::make_type<double>(), "amount",
                    ndt::make_cstruct(ndt::make_string(), "name", ndt::make_date(), "when"), "data"));

    n = parse_json(sdt, "[{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, \n"
                    "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] },\n"
                    "{\"position\":[1,2,3], \"amount\": 3.125,\n"
                    "\"data\":{ \"when\":\"2013-12-25\", \"name\":\"Frank\"}}]");
    EXPECT_EQ(3.5,          n(0,0,0).as<float>());
    EXPECT_EQ(1.0,          n(0,0,1).as<float>());
    EXPECT_EQ(1e10,         n(0,0,2).as<float>());
    EXPECT_EQ(10.5,         n(0,1).as<double>());
    EXPECT_EQ("Harvey",     n(0,2,0).as<string>());
    EXPECT_EQ("1970-02-13", n(0,2,1).as<string>());
    EXPECT_EQ(1,            n(1,0,0).as<float>());
    EXPECT_EQ(2,            n(1,0,1).as<float>());
    EXPECT_EQ(3,            n(1,0,2).as<float>());
    EXPECT_EQ(3.125,        n(1,1).as<double>());
    EXPECT_EQ("Frank",      n(1,2,0).as<string>());
    EXPECT_EQ("2013-12-25", n(1,2,1).as<string>());

    // Spurious '#' inserted
    EXPECT_THROW(parse_json(sdt, "[{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, \n"
                    "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] },\n"
                    "{\"position\":[1,2,3], \"amount\": 3.125#,\n"
                    "\"data\":{ \"when\":\"2013-12-25\", \"name\":\"Frank\"}}]"),
                    invalid_argument);
}

TEST(JSONParser, JSONDType) {
    nd::array n;

    // Parsing JSON with the output being just a json string
    n = parse_json("json", "{\"a\":3.14}");
    EXPECT_EQ(ndt::make_json(), n.get_type());
    EXPECT_EQ("{\"a\":3.14}", n.as<string>());

    // Parsing JSON with a piece of it being a json string
    n = parse_json("{a: json, b: int32, c: string}",
                    "{\"c\": \"testing string\", \"a\": [3.1, {\"X\":2}, [1,2]], \"b\":12}");
    EXPECT_EQ(ndt::make_struct(ndt::make_json(), "a", ndt::make_type<int32_t>(), "b", ndt::make_string(), "c"),
                    n.get_type());
    EXPECT_EQ("[3.1, {\"X\":2}, [1,2]]", n(0).as<string>());
    EXPECT_EQ(12, n(1).as<int>());
    EXPECT_EQ("testing string", n(2).as<string>());
}
