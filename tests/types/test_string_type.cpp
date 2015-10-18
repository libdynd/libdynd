//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/gfunc/call_gcallable.hpp>
#include <dynd/dim_iter.hpp>

using namespace std;
using namespace dynd;

TEST(StringType, Create)
{
  ndt::type d;

  // Strings with various encodings
  d = ndt::string_type::make();
  EXPECT_EQ(string_type_id, d.get_type_id());
  EXPECT_EQ(string_kind, d.get_kind());
  EXPECT_EQ(alignof(dynd::string), d.get_data_alignment());
  EXPECT_EQ(sizeof(dynd::string), d.get_data_size());
  EXPECT_FALSE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::string_type::make();
  EXPECT_EQ(string_type_id, d.get_type_id());
  EXPECT_EQ(string_kind, d.get_kind());
  EXPECT_EQ(alignof(dynd::string), d.get_data_alignment());
  EXPECT_EQ(sizeof(dynd::string), d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::string_type::make();
  EXPECT_EQ(string_type_id, d.get_type_id());
  EXPECT_EQ(string_kind, d.get_kind());
  EXPECT_EQ(alignof(dynd::string), d.get_data_alignment());
  EXPECT_EQ(sizeof(dynd::string), d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::string_type::make();
  EXPECT_EQ(string_type_id, d.get_type_id());
  EXPECT_EQ(string_kind, d.get_kind());
  EXPECT_EQ(sizeof(void *), d.get_data_alignment());
  EXPECT_EQ(sizeof(dynd::string), d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::string_type::make();
  EXPECT_EQ(string_type_id, d.get_type_id());
  EXPECT_EQ(string_kind, d.get_kind());
  EXPECT_EQ(alignof(dynd::string), d.get_data_alignment());
  EXPECT_EQ(sizeof(dynd::string), d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(StringType, ArrayCreation)
{
  nd::array a;

  // A C-style string literal
  a = "testing string construction";
  EXPECT_EQ(ndt::string_type::make(), a.get_type());
  EXPECT_EQ("testing string construction", a.as<std::string>());

  // A C-style char array variable
  const char carr[] = "string construction";
  a = carr;
  EXPECT_EQ(ndt::string_type::make(), a.get_type());
  EXPECT_EQ("string construction", a.as<std::string>());

  // A C-style char pointer variable
  const char *cptr = "construction";
  a = cptr;
  EXPECT_EQ(ndt::string_type::make(), a.get_type());
  EXPECT_EQ("construction", a.as<std::string>());

  // An array of UTF8 strings
  const char *i0[5] = {"this", "is", "a", "test", "of strings that are various sizes"};
  a = i0;
  EXPECT_EQ(ndt::type("5 * string"), a.get_type());
  EXPECT_EQ(a.get_shape()[0], 5);
  EXPECT_EQ("this", a(0).as<std::string>());
  EXPECT_EQ("is", a(1).as<std::string>());
  EXPECT_EQ("a", a(2).as<std::string>());
  EXPECT_EQ("test", a(3).as<std::string>());
  EXPECT_EQ("of strings that are various sizes", a(4).as<std::string>());
}

TEST(StringType, Basic)
{
  nd::array a, b;

  // std::string goes in as a utf8 string
  a = std::string("abcdefg");
  EXPECT_EQ(ndt::string_type::make(), a.get_type());
  EXPECT_EQ(std::string("abcdefg"), a.as<std::string>());
  // Make it a fixed_string for this test
  a = a.ucast(ndt::fixed_string_type::make(7, string_encoding_utf_8)).eval();

  // Convert to a blockref string type with the same utf8 codec
  b = a.ucast(ndt::string_type::make());
  EXPECT_EQ(ndt::convert_type::make(ndt::string_type::make(), ndt::fixed_string_type::make(7, string_encoding_utf_8)),
            b.get_type());
  b = b.eval();
  EXPECT_EQ(ndt::string_type::make(), b.get_type());
  EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());

  // Convert to a blockref string type with the utf16 codec
  b = a.ucast(ndt::string_type::make());
  EXPECT_EQ(ndt::convert_type::make(ndt::string_type::make(), ndt::fixed_string_type::make(7, string_encoding_utf_8)),
            b.get_type());
  b = b.eval();
  EXPECT_EQ(ndt::string_type::make(), b.get_type());
  EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());

  // Convert to a blockref string type with the utf32 codec
  b = a.ucast(ndt::string_type::make());
  EXPECT_EQ(ndt::convert_type::make(ndt::string_type::make(), ndt::fixed_string_type::make(7, string_encoding_utf_8)),
            b.get_type());
  b = b.eval();
  EXPECT_EQ(ndt::string_type::make(), b.get_type());
  EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());

  // Convert to a blockref string type with the ascii codec
  b = a.ucast(ndt::string_type::make());
  EXPECT_EQ(ndt::convert_type::make(ndt::string_type::make(), ndt::fixed_string_type::make(7, string_encoding_utf_8)),
            b.get_type());
  b = b.eval();
  EXPECT_EQ(ndt::string_type::make(), b.get_type());
  EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());
}

TEST(StringType, AccessFlags)
{
  nd::array a, b;

  // Default construction from a string produces an immutable fixed_string
  a = std::string("testing one two three testing one two three four five "
                  "testing one two three four five six seven");
  EXPECT_EQ(nd::read_access_flag | nd::immutable_access_flag, (int)a.get_access_flags());
  // Turn it into a fixed_string type for this test
  a = a.ucast(ndt::fixed_string_type::make(95, string_encoding_utf_8)).eval();
  EXPECT_EQ(ndt::fixed_string_type::make(95, string_encoding_utf_8), a.get_type());

  // Converting to a blockref string of the same encoding produces a reference
  // into the fixed_string value
  b = a.ucast(ndt::string_type::make()).eval();
  EXPECT_EQ(nd::read_access_flag | nd::write_access_flag, (int)b.get_access_flags());
  EXPECT_EQ(ndt::string_type::make(), b.get_type());
  // The data array for 'a' matches the referenced data for 'b' (TODO: Restore
  // this property)
  //    EXPECT_EQ(a.get_readonly_originptr(), reinterpret_cast<const char *
  //    const *>(b.get_readonly_originptr())[0]);
}

/*
TEST(StringType, Unicode)
{
  static uint32_t utf32_string[] = {0x0000, 0x0001, 0x0020,  0x007f,  0x0080,
                                    0x00ff, 0x07ff, 0x0800,  0xd7ff,  0xe000,
                                    0xfffd, 0xffff, 0x10000, 0x10ffff};
  static uint16_t utf16_string[] = {
      0x0000, 0x0001, 0x0020, 0x007f, 0x0080, 0x00ff, 0x07ff, 0x0800,
      0xd7ff, // Just below "low surrogate range"
      0xe000, // Just above "high surrogate range"
      0xfffd,
      0xffff,         // Largest utf16 1-character code point
      0xd800, 0xdc00, // Smallest utf16 2-character code point
      0xdbff, 0xdfff  // Largest code point
  };
  static uint8_t utf8_string[] = {
      0x00,             // NULL code point
      0x01,             // Smallest non-NULL code point
      0x20,             // Smallest non-control code point
      0x7f,             // Largest utf8 1-character code point
      0xc2, 0x80,       // Smallest utf8 2-character code point
      0xc3, 0xbf,       // 0xff code point
      0xdf, 0xbf,       // Largest utf8 2-character code point
      0xe0, 0xa0, 0x80, // Smallest utf8 3-character code point
      0xed, 0x9f, 0xbf, // Just below "low surrogate range"
      0xee, 0x80, 0x80, // Just above "high surrogate range"
      0xef, 0xbf, 0xbd, // 0xfffd code point
      0xef, 0xbf, 0xbf, // Largest utf8 3-character code point
      0xf0, 0x90, 0x80,
      0x80, // Smallest utf8 4-character code point
      0xf4, 0x8f, 0xbf,
      0xbf // Largest code point
  };
  nd::array x;
  nd::array a(nd::make_utf32_array(utf32_string));
  nd::array b(nd::make_utf16_array(utf16_string));
  nd::array c(nd::make_utf8_array(utf8_string));

  // Convert all to utf32 and compare with the reference
  x = a.ucast(ndt::string_type::make(string_encoding_utf_32)).eval();
  EXPECT_EQ(0, memcmp(utf32_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf32_string)));
  x = b.ucast(ndt::string_type::make(string_encoding_utf_32)).eval();
  EXPECT_EQ(0, memcmp(utf32_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf32_string)));
  x = c.ucast(ndt::string_type::make(string_encoding_utf_32)).eval();
  EXPECT_EQ(0, memcmp(utf32_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf32_string)));

  // Convert all to utf16 and compare with the reference
  x = a.ucast(ndt::string_type::make(string_encoding_utf_16)).eval();
  EXPECT_EQ(0, memcmp(utf16_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf16_string)));
  x = b.ucast(ndt::string_type::make(string_encoding_utf_16)).eval();
  EXPECT_EQ(0, memcmp(utf16_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf16_string)));
  x = c.ucast(ndt::string_type::make(string_encoding_utf_16)).eval();
  EXPECT_EQ(0, memcmp(utf16_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf16_string)));

  // Convert all to utf8 and compare with the reference
  x = a.ucast(ndt::string_type::make(string_encoding_utf_8)).eval();
  EXPECT_EQ(0, memcmp(utf8_string, *reinterpret_cast<const char *const *>(
                                       x.get_readonly_originptr()),
                      sizeof(utf8_string)));
  x = b.ucast(ndt::string_type::make(string_encoding_utf_8)).eval();
  EXPECT_EQ(0, memcmp(utf8_string, *reinterpret_cast<const char *const *>(
                                       x.get_readonly_originptr()),
                      sizeof(utf8_string)));
  x = c.ucast(ndt::string_type::make(string_encoding_utf_8)).eval();
  EXPECT_EQ(0, memcmp(utf8_string, *reinterpret_cast<const char *const *>(
                                       x.get_readonly_originptr()),
                      sizeof(utf8_string)));
}
*/

TEST(StringType, CanonicalDType)
{
  // The canonical type of a string type is the same type
  EXPECT_EQ((ndt::string_type::make()), (ndt::string_type::make().get_canonical_type()));
}

TEST(StringType, Storage)
{
  nd::array a;

  a = "testing";
  EXPECT_EQ(ndt::bytes_type::make(1), a.storage().get_type());
}

TEST(StringType, EncodingSizes)
{
  EXPECT_EQ(1, string_encoding_char_size_table[string_encoding_ascii]);
  EXPECT_EQ(1, string_encoding_char_size_table[string_encoding_utf_8]);
  EXPECT_EQ(2, string_encoding_char_size_table[string_encoding_ucs_2]);
  EXPECT_EQ(2, string_encoding_char_size_table[string_encoding_utf_16]);
  EXPECT_EQ(4, string_encoding_char_size_table[string_encoding_utf_32]);
}

TEST(StringType, StringToBool)
{
  EXPECT_TRUE(nd::array("true").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array(" True").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("TRUE ").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("T").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("yes  ").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("Yes").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("Y").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array(" on").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("On").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("1").ucast<bool1>().as<bool>());

  EXPECT_FALSE(nd::array("false").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("False").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("FALSE ").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("F ").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array(" no").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("No").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("N").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("off ").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("Off").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("0 ").ucast<bool1>().as<bool>());

  // By default, conversion to bool is not permissive
  EXPECT_THROW(nd::array(nd::array("").ucast<bool1>().eval()), invalid_argument);
  EXPECT_THROW(nd::array(nd::array("2").ucast<bool1>().eval()), invalid_argument);
  EXPECT_THROW(nd::array(nd::array("flase").ucast<bool1>().eval()), invalid_argument);

  // In "nocheck" mode, it's a bit more permissive
  eval::eval_context tmp_ectx;
  tmp_ectx.errmode = assign_error_nocheck;
  EXPECT_FALSE(nd::array(nd::array("").ucast<bool1>().eval(&tmp_ectx)).as<bool>());
  EXPECT_TRUE(nd::array(nd::array("2").ucast<bool1>().eval(&tmp_ectx)).as<bool>());
  EXPECT_TRUE(nd::array(nd::array("flase").ucast<bool1>().eval(&tmp_ectx)).as<bool>());
}

TEST(StringType, StringToInteger)
{
  // Test the boundary cases of the various integers
  EXPECT_EQ(-128, nd::array("-128").ucast<int8_t>().as<int8_t>());
  EXPECT_EQ(127, nd::array("127").ucast<int8_t>().as<int8_t>());
  EXPECT_THROW(nd::array("-129").ucast<int8_t>().eval(), runtime_error);
  EXPECT_THROW(nd::array("128").ucast<int8_t>().eval(), runtime_error);

  EXPECT_EQ(-32768, nd::array("-32768").ucast<int16_t>().as<int16_t>());
  EXPECT_EQ(32767, nd::array("32767").ucast<int16_t>().as<int16_t>());
  EXPECT_THROW(nd::array("-32769").ucast<int16_t>().eval(), runtime_error);
  EXPECT_THROW(nd::array("32768").ucast<int16_t>().eval(), runtime_error);

  EXPECT_EQ(-2147483648LL, nd::array("-2147483648").ucast<int32_t>().as<int32_t>());
  EXPECT_EQ(2147483647, nd::array("2147483647").ucast<int32_t>().as<int32_t>());
  EXPECT_THROW(nd::array(nd::array("-2147483649").ucast<int32_t>().eval()), runtime_error);
  EXPECT_THROW(nd::array(nd::array("2147483648").ucast<int32_t>().eval()), runtime_error);

  EXPECT_EQ(-9223372036854775807LL - 1, nd::array("-9223372036854775808").ucast<int64_t>().as<int64_t>());
  EXPECT_EQ(9223372036854775807LL, nd::array("9223372036854775807").ucast<int64_t>().as<int64_t>());
  EXPECT_THROW(nd::array("-9223372036854775809").ucast<int64_t>().eval(), runtime_error);
  EXPECT_THROW(nd::array("9223372036854775808").ucast<int64_t>().eval(), runtime_error);

  EXPECT_EQ(0u, nd::array("0").ucast<uint8_t>().as<uint8_t>());
  EXPECT_EQ(255u, nd::array("255").ucast<uint8_t>().as<uint8_t>());
  EXPECT_THROW(nd::array("-1").ucast<uint8_t>().eval(), runtime_error);
  EXPECT_THROW(nd::array("256").ucast<uint8_t>().eval(), runtime_error);

  EXPECT_EQ(0u, nd::array("0").ucast<uint16_t>().as<uint16_t>());
  EXPECT_EQ(65535u, nd::array("65535").ucast<uint16_t>().as<uint16_t>());
  EXPECT_THROW(nd::array("-1").ucast<uint16_t>().eval(), runtime_error);
  EXPECT_THROW(nd::array("65536").ucast<uint16_t>().eval(), runtime_error);

  EXPECT_EQ(0u, nd::array("0").ucast<uint32_t>().as<uint32_t>());
  EXPECT_EQ(4294967295ULL, nd::array("4294967295").ucast<uint32_t>().as<uint32_t>());
  EXPECT_THROW(nd::array("-1").ucast<uint32_t>().eval(), runtime_error);
  EXPECT_THROW(nd::array("4294967296").ucast<uint32_t>().eval(), runtime_error);

  EXPECT_EQ(0u, nd::array("0").ucast<uint64_t>().as<uint64_t>());
  EXPECT_EQ(18446744073709551615ULL, nd::array("18446744073709551615").ucast<uint64_t>().as<uint64_t>());
  EXPECT_THROW(nd::array("-1").ucast<uint64_t>().eval(), runtime_error);
  EXPECT_THROW(nd::array("18446744073709551616").ucast<uint64_t>().eval(), runtime_error);

  EXPECT_THROW(nd::array("").ucast<uint64_t>().eval(), invalid_argument);
  EXPECT_THROW(nd::array("-").ucast<uint64_t>().eval(), invalid_argument);
}

TEST(StringType, StringToFloat32SpecialValues)
{
  // +NaN with default payload
  EXPECT_EQ(0x7fc00000u, nd::array("NaN").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  EXPECT_EQ(0x7fc00000u, nd::array("nan").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  EXPECT_EQ(0x7fc00000u, nd::array("1.#QNAN").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  // -NaN with default payload
  EXPECT_EQ(0xffc00000u, nd::array("-NaN").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  EXPECT_EQ(0xffc00000u, nd::array("-nan").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  EXPECT_EQ(0xffc00000u, nd::array("-1.#IND").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  // +Inf
  EXPECT_EQ(0x7f800000u, nd::array("Inf").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  EXPECT_EQ(0x7f800000u, nd::array("inf").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  EXPECT_EQ(0x7f800000u, nd::array("Infinity").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  EXPECT_EQ(0x7f800000u, nd::array("1.#INF").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  // -Inf
  EXPECT_EQ(0xff800000u, nd::array("-Inf").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  EXPECT_EQ(0xff800000u, nd::array("-inf").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  EXPECT_EQ(0xff800000u, nd::array("-Infinity").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
  EXPECT_EQ(0xff800000u, nd::array("-1.#INF").ucast<float>().view_scalars<uint32_t>().as<uint32_t>());
}

TEST(StringType, StringToFloat64SpecialValues)
{
  // +NaN with default payload
  EXPECT_EQ(0x7ff8000000000000ULL, nd::array("NaN").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  EXPECT_EQ(0x7ff8000000000000ULL, nd::array("nan").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  EXPECT_EQ(0x7ff8000000000000ULL, nd::array("1.#QNAN").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  // -NaN with default payload
  EXPECT_EQ(0xfff8000000000000ULL, nd::array("-NaN").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  EXPECT_EQ(0xfff8000000000000ULL, nd::array("-nan").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  EXPECT_EQ(0xfff8000000000000ULL, nd::array("-1.#IND").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  // +Inf
  EXPECT_EQ(0x7ff0000000000000ULL, nd::array("Inf").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  EXPECT_EQ(0x7ff0000000000000ULL, nd::array("inf").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  EXPECT_EQ(0x7ff0000000000000ULL, nd::array("Infinity").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  EXPECT_EQ(0x7ff0000000000000ULL, nd::array("1.#INF").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  // -Inf
  EXPECT_EQ(0xfff0000000000000ULL, nd::array("-Inf").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  EXPECT_EQ(0xfff0000000000000ULL, nd::array("-inf").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  EXPECT_EQ(0xfff0000000000000ULL, nd::array("-Infinity").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
  EXPECT_EQ(0xfff0000000000000ULL, nd::array("-1.#INF").ucast<double>().view_scalars<uint64_t>().as<uint64_t>());
}

TEST(StringType, Comparisons)
{
  nd::array a, b;

  // Basic test
  a = nd::array("abc");
  b = nd::array("abd");
  //    EXPECT_TRUE(a.op_sorting_less(b));
  EXPECT_TRUE(static_cast<bool>(a < b));
  EXPECT_TRUE(static_cast<bool>(a <= b));
  EXPECT_FALSE(static_cast<bool>(a == b));
  EXPECT_TRUE(static_cast<bool>(a != b));
  EXPECT_FALSE(static_cast<bool>(a >= b));
  EXPECT_FALSE(static_cast<bool>(a > b));
  //  EXPECT_FALSE(b.op_sorting_less(a));
  EXPECT_FALSE(static_cast<bool>(b < a));
  EXPECT_FALSE(static_cast<bool>(b <= a));
  EXPECT_FALSE(static_cast<bool>(b == a));
  EXPECT_TRUE(static_cast<bool>(b != a));
  EXPECT_TRUE(static_cast<bool>(b >= a));
  EXPECT_TRUE(static_cast<bool>(b > a));

  // Different sizes
  a = nd::array("abcd");
  b = nd::array("abcde");
  //  EXPECT_TRUE(a.op_sorting_less(b));
  EXPECT_TRUE(static_cast<bool>(a < b));
  EXPECT_TRUE(static_cast<bool>(a <= b));
  EXPECT_FALSE(static_cast<bool>(a == b));
  EXPECT_TRUE(static_cast<bool>(a != b));
  EXPECT_FALSE(static_cast<bool>(a >= b));
  EXPECT_FALSE(static_cast<bool>(a > b));
  //  EXPECT_FALSE(b.op_sorting_less(a));
  EXPECT_FALSE(static_cast<bool>(b < a));
  EXPECT_FALSE(static_cast<bool>(b <= a));
  EXPECT_FALSE(static_cast<bool>(b == a));
  EXPECT_TRUE(static_cast<bool>(b != a));
  EXPECT_TRUE(static_cast<bool>(b >= a));
  EXPECT_TRUE(static_cast<bool>(b > a));

  // Expression and different encodings
  /*
  a = nd::array("abcd").ucast(ndt::string_type::make(string_encoding_ucs_2));
  b = nd::array("abcde").ucast(ndt::string_type::make(string_encoding_utf_32)).eval();
        EXPECT_TRUE(a.op_sorting_less(b));
        EXPECT_TRUE(a < b);
        EXPECT_TRUE(a <= b);
        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);
        EXPECT_FALSE(a >= b);
        EXPECT_FALSE(a > b);
        EXPECT_FALSE(b.op_sorting_less(a));
        EXPECT_FALSE(b < a);
        EXPECT_FALSE(b <= a);
        EXPECT_FALSE(b == a);
        EXPECT_TRUE(b != a);
        EXPECT_TRUE(b >= a);
        EXPECT_TRUE(b > a);
    */
}

/*
TODO: Reenable this.

TEST(StringType, Concatenation) {
    nd::array a, b, c;

    a = "first";
    b = "second";
    c = a + b;
    EXPECT_EQ("firstsecond", c.as<string>());

    const char *a_arr[3] = {"testing", "one", "two"};
    const char *b_arr[3] = {"alpha", "beta", "gamma"};

    a = a_arr;
    b = b_arr;
    c = (a + b).eval();
    ASSERT_EQ(ndt::type("3 * string"), c.get_type());
    EXPECT_EQ(3, c.get_dim_size());
    EXPECT_EQ("testingalpha", c(0).as<string>());
    EXPECT_EQ("onebeta", c(1).as<string>());
    EXPECT_EQ("twogamma", c(2).as<string>());
}
*/

/*
TEST(StringType, Find1) {
    nd::array a, b, c;

    const char *a_arr[4] = {"abc", "ababc", "ababab", "abd"};
    a = a_arr;
    b = "abc";

    c = a.f("find", b).eval();
    ASSERT_EQ(ndt::type("4 * intptr"), c.get_type());
    ASSERT_EQ(4, c.get_shape()[0]);
    EXPECT_EQ(0, c(0).as<intptr_t>());
    EXPECT_EQ(2, c(1).as<intptr_t>());
    EXPECT_EQ(-1, c(2).as<intptr_t>());
    EXPECT_EQ(-1, c(3).as<intptr_t>());
}

TEST(StringType, Find2) {
    nd::array a, b, c;

    const char *b_arr[6] = {"a", "b", "c", "bc", "d", "cd"};
    a = "abc";
    b = b_arr;

    c = a.f("find", b).eval();
    ASSERT_EQ(ndt::type("6 * intptr"), c.get_type());
    ASSERT_EQ(6, c.get_shape()[0]);
    EXPECT_EQ(0, c(0).as<intptr_t>());
    EXPECT_EQ(1, c(1).as<intptr_t>());
    EXPECT_EQ(2, c(2).as<intptr_t>());
    EXPECT_EQ(1, c(3).as<intptr_t>());
    EXPECT_EQ(-1, c(4).as<intptr_t>());
    EXPECT_EQ(-1, c(5).as<intptr_t>());
}
*/

template <class T>
static bool ascii_T_compare(const char *x, const T *y, intptr_t count)
{
  for (intptr_t i = 0; i < count; ++i) {
    if ((T)x[i] != y[i]) {
      return false;
    }
  }
  return true;
}

/*
TEST(StringType, Iter)
{
  const char *str = "This is a string for testing";
  nd::array a = str;

  dim_iter it;
  static_cast<const ndt::base_string_type *>(a.get_dtype().extended())->make_string_iter(
      &it, string_encoding_utf_8, a.get_arrmeta(), a.get_readonly_originptr(), a.get_data_memblock());
  // With a short string like this, the entire string will be
  // provided in one go
  ASSERT_EQ(1, it.vtable->next(&it));
  ASSERT_EQ((intptr_t)strlen(str), it.data_elcount);
  EXPECT_EQ(0, memcmp(str, it.data_ptr, it.data_elcount));
  it.destroy();

  static_cast<const ndt::base_string_type *>(a.get_dtype().extended())->make_string_iter(
      &it, string_encoding_utf_16, a.get_arrmeta(), a.get_readonly_originptr(), a.get_data_memblock());
  // With a short string like this, the entire string will be
  // provided in one go
  ASSERT_EQ(1, it.vtable->next(&it));
  ASSERT_EQ((intptr_t)strlen(str), it.data_elcount);
  EXPECT_TRUE(ascii_T_compare(str, reinterpret_cast<const uint16_t *>(it.data_ptr), it.data_elcount));
  it.destroy();

  static_cast<const ndt::base_string_type *>(a.get_dtype().extended())->make_string_iter(
      &it, string_encoding_utf_32, a.get_arrmeta(), a.get_readonly_originptr(), a.get_data_memblock());
  // With a short string like this, the entire string will be
  // provided in one go
  ASSERT_EQ(1, it.vtable->next(&it));
  ASSERT_EQ((intptr_t)strlen(str), it.data_elcount);
  EXPECT_TRUE(ascii_T_compare(str, reinterpret_cast<const uint32_t *>(it.data_ptr), it.data_elcount));
  it.destroy();
}
*/
