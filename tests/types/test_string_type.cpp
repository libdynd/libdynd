//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/string.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

TEST(StringType, Create) {
  ndt::type d;

  // Strings with various encodings
  d = ndt::make_type<ndt::string_type>();
  EXPECT_EQ(string_id, d.get_id());
  EXPECT_EQ(string_kind_id, d.get_base_id());
  EXPECT_LE(sizeof(void *), d.get_data_alignment());
  EXPECT_EQ(alignof(dynd::string), d.get_data_alignment());
  EXPECT_EQ(16u, sizeof(dynd::string));
  EXPECT_EQ(sizeof(dynd::string), d.get_data_size());
  EXPECT_FALSE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(StringType, ArrayCreation) {
  nd::array a;

  // A C-style string literal
  a = "testing string construction";
  EXPECT_EQ(ndt::make_type<ndt::string_type>(), a.get_type());
  EXPECT_EQ("testing string construction", a.as<std::string>());

  // A C-style char array variable
  const char carr[] = "string construction";
  a = carr;
  EXPECT_EQ(ndt::make_type<ndt::string_type>(), a.get_type());
  EXPECT_EQ("string construction", a.as<std::string>());

  // A C-style char pointer variable
  const char *cptr = "construction";
  a = cptr;
  EXPECT_EQ(ndt::make_type<ndt::string_type>(), a.get_type());
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

TEST(StringType, AccessFlags) {
  nd::array a, b;

  // Default construction from a string produces an immutable fixed_string
  a = std::string("testing one two three testing one two three four five "
                  "testing one two three four five six seven");
  //  EXPECT_EQ(nd::read_access_flag | nd::immutable_access_flag, (int)a.get_flags());
  // Turn it into a fixed_string type for this test
  a = nd::empty(ndt::make_type<ndt::fixed_string_type>(95, string_encoding_utf_8)).assign(a);
  EXPECT_EQ(ndt::make_type<ndt::fixed_string_type>(95, string_encoding_utf_8), a.get_type());

  // Converting to a blockref string of the same encoding produces a reference
  // into the fixed_string value
  b = nd::empty(ndt::make_type<ndt::string_type>()).assign(a);
  EXPECT_EQ(nd::read_access_flag | nd::write_access_flag, (int)b.get_flags());
  EXPECT_EQ(ndt::make_type<ndt::string_type>(), b.get_type());
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
  x = a.ucast(ndt::make_type<ndt::string_type>(string_encoding_utf_32)).eval();
  EXPECT_EQ(0, memcmp(utf32_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf32_string)));
  x = b.ucast(ndt::make_type<ndt::string_type>(string_encoding_utf_32)).eval();
  EXPECT_EQ(0, memcmp(utf32_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf32_string)));
  x = c.ucast(ndt::make_type<ndt::string_type>(string_encoding_utf_32)).eval();
  EXPECT_EQ(0, memcmp(utf32_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf32_string)));

  // Convert all to utf16 and compare with the reference
  x = a.ucast(ndt::make_type<ndt::string_type>(string_encoding_utf_16)).eval();
  EXPECT_EQ(0, memcmp(utf16_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf16_string)));
  x = b.ucast(ndt::make_type<ndt::string_type>(string_encoding_utf_16)).eval();
  EXPECT_EQ(0, memcmp(utf16_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf16_string)));
  x = c.ucast(ndt::make_type<ndt::string_type>(string_encoding_utf_16)).eval();
  EXPECT_EQ(0, memcmp(utf16_string, *reinterpret_cast<const char *const *>(
                                        x.get_readonly_originptr()),
                      sizeof(utf16_string)));

  // Convert all to utf8 and compare with the reference
  x = a.ucast(ndt::make_type<ndt::string_type>(string_encoding_utf_8)).eval();
  EXPECT_EQ(0, memcmp(utf8_string, *reinterpret_cast<const char *const *>(
                                       x.get_readonly_originptr()),
                      sizeof(utf8_string)));
  x = b.ucast(ndt::make_type<ndt::string_type>(string_encoding_utf_8)).eval();
  EXPECT_EQ(0, memcmp(utf8_string, *reinterpret_cast<const char *const *>(
                                       x.get_readonly_originptr()),
                      sizeof(utf8_string)));
  x = c.ucast(ndt::make_type<ndt::string_type>(string_encoding_utf_8)).eval();
  EXPECT_EQ(0, memcmp(utf8_string, *reinterpret_cast<const char *const *>(
                                       x.get_readonly_originptr()),
                      sizeof(utf8_string)));
}
*/

TEST(StringType, CanonicalDType) {
  // The canonical type of a string type is the same type
  EXPECT_EQ((ndt::make_type<ndt::string_type>()), (ndt::make_type<ndt::string_type>().get_canonical_type()));
}

TEST(StringType, Storage) {
  nd::array a;

  a = "testing";
  EXPECT_EQ(ndt::make_type<ndt::bytes_type>(1), a.storage().get_type());
}

TEST(StringType, Properties) {
  ndt::type d = ndt::make_type<ndt::string_type>();

  EXPECT_EQ("utf8", d.p<std::string>("encoding"));
}

TEST(StringType, EncodingSizes) {
  EXPECT_EQ(1, string_encoding_char_size_table[string_encoding_ascii]);
  EXPECT_EQ(1, string_encoding_char_size_table[string_encoding_utf_8]);
  EXPECT_EQ(2, string_encoding_char_size_table[string_encoding_ucs_2]);
  EXPECT_EQ(2, string_encoding_char_size_table[string_encoding_utf_16]);
  EXPECT_EQ(4, string_encoding_char_size_table[string_encoding_utf_32]);
}

/*
  ToDo: Reenable this.

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
*/

TEST(StringType, StringToInteger) {
  // Test the boundary cases of the various integers
  nd::array i8 = nd::empty(ndt::make_type<int8_t>());
  EXPECT_EQ(-128, i8.assign(nd::array("-128")).as<int8_t>());
  EXPECT_EQ(127, i8.assign(nd::array("127")).as<int8_t>());
  EXPECT_THROW(i8.assign(nd::array("-129")), runtime_error);
  EXPECT_THROW(i8.assign(nd::array("128")), runtime_error);

  nd::array i16 = nd::empty(ndt::make_type<int16_t>());
  EXPECT_EQ(-32768, i16.assign(nd::array("-32768")).as<int16_t>());
  EXPECT_EQ(32767, i16.assign(nd::array("32767")).as<int16_t>());
  EXPECT_THROW(i16.assign(nd::array("-32769")), runtime_error);
  EXPECT_THROW(i16.assign(nd::array("32768")), runtime_error);

  nd::array i32 = nd::empty(ndt::make_type<int32_t>());
  EXPECT_EQ(-2147483648LL, i32.assign(nd::array("-2147483648")).as<int32_t>());
  EXPECT_EQ(2147483647, i32.assign(nd::array("2147483647")).as<int32_t>());
  EXPECT_THROW(i32.assign(nd::array("-2147483649")), runtime_error);
  EXPECT_THROW(i32.assign(nd::array("2147483648")), runtime_error);

  /*
    ToDo: Reenable this.

    EXPECT_EQ(-9223372036854775807LL - 1, nd::array("-9223372036854775808").ucast<int64_t>().as<int64_t>());
    EXPECT_EQ(9223372036854775807LL, nd::array("9223372036854775807").ucast<int64_t>().as<int64_t>());
    EXPECT_THROW(nd::array("-9223372036854775809").ucast<int64_t>().eval(), runtime_error);
    EXPECT_THROW(nd::array("9223372036854775808").ucast<int64_t>().eval(), runtime_error);
  */

  /*
    nd::array u8 = nd::empty(ndt::make_type<uint8_t>());
    EXPECT_EQ(0u, u8.assign(nd::array("0")).as<uint8_t>());
    EXPECT_EQ(255u, u8.assign(nd::array("255")).as<uint8_t>());
    EXPECT_THROW(u8.assign(nd::array("-1")), invalid_argument);
    EXPECT_THROW(u8.assign(nd::array("256")), out_of_range);

    nd::array u16 = nd::empty(ndt::make_type<uint16_t>());
    EXPECT_EQ(0u, u16.assign(nd::array("0")).as<uint16_t>());
    EXPECT_EQ(65535u, u16.assign(nd::array("65535")).as<uint16_t>());
    EXPECT_THROW(u16.assign(nd::array("-1")), invalid_argument);
    EXPECT_THROW(u16.assign(nd::array("65536")), out_of_range);

    nd::array u32 = nd::empty(ndt::make_type<uint32_t>());
    EXPECT_EQ(0u, u32.assign(nd::array("0")).as<uint32_t>());
    EXPECT_EQ(4294967295ULL, u32.assign(nd::array("4294967295")).as<uint32_t>());
    EXPECT_THROW(u32.assign(nd::array("-1")), invalid_argument);
    EXPECT_THROW(u32.assign(nd::array("4294967296")), out_of_range);

    nd::array u64 = nd::empty(ndt::make_type<uint64_t>());
    EXPECT_EQ(0u, u64.assign(nd::array("0")).as<uint64_t>());
    EXPECT_EQ(18446744073709551615ULL, u64.assign(nd::array("18446744073709551615")).as<uint64_t>());
    EXPECT_THROW(u64.assign(nd::array("-1")), invalid_argument);
    EXPECT_THROW(u64.assign(nd::array("18446744073709551616")), out_of_range);

  EXPECT_THROW(u64.assign(nd::array("")), invalid_argument);
  EXPECT_THROW(u64.assign(nd::array("-")), invalid_argument);
  */
}

TEST(StringType, Comparisons) {
  nd::array a, b;

  // Basic test
  a = nd::array("abc");
  b = nd::array("abd");
  //    EXPECT_TRUE(a.op_sorting_less(b));
  EXPECT_TRUE((a < b).as<bool>());
  EXPECT_TRUE((a <= b).as<bool>());
  EXPECT_FALSE((a == b).as<bool>());
  EXPECT_TRUE((a != b).as<bool>());
  EXPECT_FALSE((a >= b).as<bool>());
  EXPECT_FALSE((a > b).as<bool>());
  //  EXPECT_FALSE(b.op_sorting_less(a));
  EXPECT_FALSE((b < a).as<bool>());
  EXPECT_FALSE((b <= a).as<bool>());
  EXPECT_FALSE((b == a).as<bool>());
  EXPECT_TRUE((b != a).as<bool>());
  EXPECT_TRUE((b >= a).as<bool>());
  EXPECT_TRUE((b > a).as<bool>());

  // Different sizes
  a = nd::array("abcd");
  b = nd::array("abcde");
  //  EXPECT_TRUE(a.op_sorting_less(b));
  EXPECT_TRUE((a < b).as<bool>());
  EXPECT_TRUE((a <= b).as<bool>());
  EXPECT_FALSE((a == b).as<bool>());
  EXPECT_TRUE((a != b).as<bool>());
  EXPECT_FALSE((a >= b).as<bool>());
  EXPECT_FALSE((a > b).as<bool>());
  //  EXPECT_FALSE(b.op_sorting_less(a));
  EXPECT_FALSE((b < a).as<bool>());
  EXPECT_FALSE((b <= a).as<bool>());
  EXPECT_FALSE((b == a).as<bool>());
  EXPECT_TRUE((b != a).as<bool>());
  EXPECT_TRUE((b >= a).as<bool>());
  EXPECT_TRUE((b > a).as<bool>());

  // Expression and different encodings
  /*
  a = nd::array("abcd").ucast(ndt::make_type<ndt::string_type>(string_encoding_ucs_2));
  b = nd::array("abcde").ucast(ndt::make_type<ndt::string_type>(string_encoding_utf_32)).eval();
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

TEST(StringType, ConcatenationScalar) {
  dynd::string a("first");
  dynd::string b("second");

  dynd::string c(a + b);
  ASSERT_EQ(dynd::string("firstsecond"), c);

  a = dynd::string("foo");
  a += dynd::string("bar");

  ASSERT_EQ(dynd::string("foobar"), a);
}

TEST(StringType, Concatenation) {
  nd::array a, b, c;

  a = "first";
  b = "second";
  c = nd::string_concatenation(a, b);
  EXPECT_ARRAY_EQ("firstsecond", c);

  // c = a + b;
  // EXPECT_ARRAY_EQ("firstsecond", c);

  a = {"testing", "one", "two"};
  b = {"alpha", "beta", "gamma"};
  EXPECT_ARRAY_EQ(nd::array({"testingalpha", "onebeta", "twogamma"}), nd::string_concatenation(a, b));
}

TEST(StringType, Find1) {
  nd::array a, b;

  a = {"abc", "ababc", "ababab", "abd"};
  b = "abc";
  intptr_t c[] = {0, 2, -1, -1};

  EXPECT_ARRAY_EQ(c, nd::string_find(a, b));
}

TEST(StringType, Find2) {
  nd::array a, b;

  a = "abc";
  b = {"a", "b", "c", "bc", "d", "cd"};
  intptr_t c[] = {0, 1, 2, 1, -1, -1};

  EXPECT_ARRAY_EQ(c, nd::string_find(a, b));
}

TEST(StringType, Find3) {
  /* This tests the "fast path" where the needle is a single
     character */
  nd::array a, b;

  a = {"a", "bbbb", "bbbba", "0123456789bb", "0123456789a"};
  b = "a";
  intptr_t c[] = {0, -1, 4, -1, 10};

  EXPECT_ARRAY_EQ(c, nd::string_find(a, b));
}

TEST(StringType, RFind1) {
  nd::array a, b;

  a = {"abc", "ababc", "abcdabc", "abd"};
  b = "abc";
  intptr_t c[] = {0, 2, 4, -1};

  EXPECT_ARRAY_EQ(c, nd::string_rfind(a, b));
}

TEST(StringType, Count1) {
  nd::array a, b;

  a = {"abc", "xxxabcxxxabcxxx", "ababab", "abd"};
  b = "abc";
  intptr_t c[] = {1, 2, 0, 0};

  EXPECT_ARRAY_EQ(c, nd::string_count(a, b));
}

TEST(StringType, Count2) {
  nd::array a, b;

  a = "abc";
  b = {"a", "b", "c", "bc", "d", "cd"};
  intptr_t c[] = {1, 1, 1, 1, 0, 0};

  EXPECT_ARRAY_EQ(c, nd::string_count(a, b));
}

TEST(StringType, Count3) {
  /* This tests the "fast path" where the needle is a single
     character */
  nd::array a, b;

  a = {"a", "baab", "0123456789bb", "0123456789aaa"};
  b = "a";
  intptr_t c[] = {1, 2, 0, 3};

  EXPECT_ARRAY_EQ(c, nd::string_count(a, b));
}

TEST(StringType, Replace) {
  nd::array a, b, c, d;

  /* Tests the five main code paths:
       - old is length 0
       - old and new are length 1
       - old and new are same length
       - old.size() > new.size()
       - old.size() < new.size()
  */

  a = {"xaxxbxxxc", "xxxabcxxxabcxxx", "cabababc", "cabababc", "foobar"};
  b = {"x", "abc", "ab", "ab", ""};
  c = {"y", "ABC", "aabb", "a", "x"};

  EXPECT_ARRAY_EQ(nd::array({"yayybyyyc", "xxxABCxxxABCxxx", "caabbaabbaabbc", "caaac", "foobar"}),
                  nd::string_replace(a, b, c));
}

TEST(StringType, Split) {
  nd::array a, b, c;

  a = {"xaxxbxxxc", "xxxabcxxxabcxxx", "cabababc", "foobar"};
  b = {"x", "abc", "ab", ""};

  c = nd::string_split(a, b);

  EXPECT_EQ(1u, c(0).get_shape().size());
  EXPECT_EQ(7, c(0).get_shape()[0]);
  EXPECT_EQ("", c(0)(0));
  EXPECT_EQ("a", c(0)(1));
  EXPECT_EQ("", c(0)(2));
  EXPECT_EQ("b", c(0)(3));
  EXPECT_EQ("", c(0)(4));
  EXPECT_EQ("", c(0)(5));
  EXPECT_EQ("c", c(0)(6));

  EXPECT_EQ(1u, c(1).get_shape().size());
  EXPECT_EQ(3, c(1).get_shape()[0]);
  EXPECT_EQ("xxx", c(1)(0));
  EXPECT_EQ("xxx", c(1)(1));
  EXPECT_EQ("xxx", c(1)(2));

  EXPECT_EQ(1u, c(2).get_shape().size());
  EXPECT_EQ(4, c(2).get_shape()[0]);
  EXPECT_EQ("c", c(2)(0));
  EXPECT_EQ("", c(2)(1));
  EXPECT_EQ("", c(2)(2));
  EXPECT_EQ("c", c(2)(3));

  EXPECT_EQ(1u, c(3).get_shape().size());
  EXPECT_EQ(1, c(3).get_shape()[0]);
  EXPECT_EQ("foobar", c(3)(0));
}

TEST(StringType, StartsWith) {
  nd::array a, b, c;

  a = {"abcdef", "ab", "abd", "", "Xabc"};
  b = {"abc", "abc", "abc", "abc", "abc"};
  c = {true, false, false, false, false};

  EXPECT_ARRAY_EQ(c, nd::string_startswith(a, b));
}

TEST(StringType, EndsWith) {
  nd::array a, b, c;

  a = {"defabc", "ab", "abd", "", "Xabc"};
  b = {"abc", "abc", "abc", "abc", "abc"};
  c = {true, false, false, false, true};

  EXPECT_ARRAY_EQ(c, nd::string_endswith(a, b));
}

TEST(StringType, Contains) {
  nd::array a, b, c;

  a = {"defabc", "ab", "abd", "XXabcXX", "Xabc"};
  b = {"abc", "abc", "abc", "abc", "abc"};
  c = {true, false, false, true, true};

  EXPECT_ARRAY_EQ(c, nd::string_contains(a, b));
}

template <class T>
static bool ascii_T_compare(const char *x, const T *y, intptr_t count) {
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

TEST(StringType, IDOf) { EXPECT_EQ(string_id, ndt::id_of<ndt::string_type>::value); }
