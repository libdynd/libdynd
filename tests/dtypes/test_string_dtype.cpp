//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndarray.hpp>
#include <dynd/nodes/immutable_scalar_node.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(StringDType, Create) {
    dtype d;

    // Strings with various encodings
    d = make_string_dtype(string_encoding_utf_8);
    EXPECT_EQ(string_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(sizeof(void *), d.alignment());
    EXPECT_EQ(2*sizeof(void *), d.element_size());

    d = make_string_dtype(string_encoding_utf_8);
    EXPECT_EQ(string_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(sizeof(void *), d.alignment());
    EXPECT_EQ(2*sizeof(void *), d.element_size());

    d = make_string_dtype(string_encoding_ascii);
    EXPECT_EQ(string_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(sizeof(void *), d.alignment());
    EXPECT_EQ(2*sizeof(void *), d.element_size());

    d = make_string_dtype(string_encoding_utf_16);
    EXPECT_EQ(string_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(sizeof(void *), d.alignment());
    EXPECT_EQ(2*sizeof(void *), d.element_size());

    d = make_string_dtype(string_encoding_utf_32);
    EXPECT_EQ(string_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(sizeof(void *), d.alignment());
    EXPECT_EQ(2*sizeof(void *), d.element_size());
}

TEST(StringDType, Basic) {
    ndarray a, b;

    // std::string goes in as a utf8 string
    a = std::string("abcdefg");
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8), a.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), a.as<std::string>());
    // Make it a fixedstring for this test
    a = a.as_dtype(make_fixedstring_dtype(string_encoding_utf_8, 7)).vals();

    // Convert to a blockref string dtype with the same utf8 codec
    b = a.as_dtype(make_string_dtype(string_encoding_utf_8));
    EXPECT_EQ(make_convert_dtype(make_string_dtype(string_encoding_utf_8), make_fixedstring_dtype(string_encoding_utf_8, 7)),
                b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8),
                    b.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());

    // Convert to a blockref string dtype with the utf16 codec
    b = a.as_dtype(make_string_dtype(string_encoding_utf_16));
    EXPECT_EQ(make_convert_dtype(make_string_dtype(string_encoding_utf_16), make_fixedstring_dtype(string_encoding_utf_8, 7)),
                b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_string_dtype(string_encoding_utf_16),
                    b.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());

    // Convert to a blockref string dtype with the utf32 codec
    b = a.as_dtype(make_string_dtype(string_encoding_utf_32));
    EXPECT_EQ(make_convert_dtype(make_string_dtype(string_encoding_utf_32), make_fixedstring_dtype(string_encoding_utf_8, 7)),
                b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_string_dtype(string_encoding_utf_32),
                    b.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());

    // Convert to a blockref string dtype with the ascii codec
    b = a.as_dtype(make_string_dtype(string_encoding_ascii));
    EXPECT_EQ(make_convert_dtype(make_string_dtype(string_encoding_ascii), make_fixedstring_dtype(string_encoding_utf_8, 7)),
                b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_string_dtype(string_encoding_ascii),
                    b.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());
}

TEST(StringDType, AccessFlags) {
    ndarray a, b;

    // Default construction from a string produces an immutable fixedstring
    a = std::string("testing one two three testing one two three four five testing one two three four five six seven");
    EXPECT_EQ(read_access_flag | immutable_access_flag, a.get_access_flags());
    // Turn it into a fixedstring dtype for this test
    a = a.as_dtype(make_fixedstring_dtype(string_encoding_utf_8, 95)).vals();
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf_8, 95), a.get_dtype());

    // Converting to a blockref string of the same encoding produces a reference
    // into the fixedstring value
    b = a.as_dtype(make_string_dtype(string_encoding_utf_8)).vals();
    EXPECT_EQ(read_access_flag | write_access_flag, b.get_access_flags());
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8), b.get_dtype());
    // The data array for 'a' matches the referenced data for 'b'
    EXPECT_EQ(a.get_readonly_originptr(), reinterpret_cast<const char * const *>(b.get_readonly_originptr())[0]);

    // Converting to a blockref string of a different encoding makes a new
    // copy, so gets read write access
    b = a.as_dtype(make_string_dtype(string_encoding_utf_16)).vals();
    EXPECT_EQ(read_access_flag | write_access_flag, b.get_access_flags());
    EXPECT_EQ(make_string_dtype(string_encoding_utf_16), b.get_dtype());
}

TEST(StringDType, Unicode) {
    static uint32_t utf32_string[] = {
            0x0000,
            0x0001,
            0x0020,
            0x007f,
            0x0080,
            0x00ff,
            0x07ff,
            0x0800,
            0xd7ff,
            0xe000,
            0xfffd,
            0xffff,
            0x10000,
            0x10ffff
            };
    static uint16_t utf16_string[] = {
            0x0000,
            0x0001,
            0x0020,
            0x007f,
            0x0080,
            0x00ff,
            0x07ff,
            0x0800,
            0xd7ff, // Just below "low surrogate range"
            0xe000, // Just above "high surrogate range"
            0xfffd,
            0xffff, // Largest utf16 1-character code point
            0xd800, 0xdc00, // Smallest utf16 2-character code point
            0xdbff, 0xdfff // Largest code point
            };
    static uint8_t utf8_string[] = {
            0x00, // NULL code point
            0x01, // Smallest non-NULL code point
            0x20, // Smallest non-control code point
            0x7f, // Largest utf8 1-character code point
            0xc2, 0x80, // Smallest utf8 2-character code point
            0xc3, 0xbf, // 0xff code point
            0xdf, 0xbf, // Largest utf8 2-character code point
            0xe0, 0xa0, 0x80, // Smallest utf8 3-character code point
            0xed, 0x9f, 0xbf, // Just below "low surrogate range"
            0xee, 0x80, 0x80, // Just above "high surrogate range"
            0xef, 0xbf, 0xbd, // 0xfffd code point
            0xef, 0xbf, 0xbf, // Largest utf8 3-character code point
            0xf0, 0x90, 0x80, 0x80, // Smallest utf8 4-character code point
            0xf4, 0x8f, 0xbf, 0xbf // Largest code point
            };
    ndarray x;
    ndarray a(make_static_utf32_string_immutable_scalar_node(utf32_string));
    ndarray b(make_static_utf16_string_immutable_scalar_node(utf16_string));
    ndarray c(make_static_utf8_string_immutable_scalar_node(utf8_string));

    // Convert all to utf32 and compare with the reference
    x = a.as_dtype(make_string_dtype(string_encoding_utf_32)).vals();
    EXPECT_EQ(0, memcmp(utf32_string,
                *reinterpret_cast<const char * const *>(x.get_readonly_originptr()),
                sizeof(utf32_string)));
    x = b.as_dtype(make_string_dtype(string_encoding_utf_32)).vals();
    EXPECT_EQ(0, memcmp(utf32_string,
                *reinterpret_cast<const char * const *>(x.get_readonly_originptr()),
                sizeof(utf32_string)));
    x = c.as_dtype(make_string_dtype(string_encoding_utf_32)).vals();
    EXPECT_EQ(0, memcmp(utf32_string,
                *reinterpret_cast<const char * const *>(x.get_readonly_originptr()),
                sizeof(utf32_string)));

    // Convert all to utf16 and compare with the reference
    x = a.as_dtype(make_string_dtype(string_encoding_utf_16)).vals();
    EXPECT_EQ(0, memcmp(utf16_string,
                *reinterpret_cast<const char * const *>(x.get_readonly_originptr()),
                sizeof(utf16_string)));
    x = b.as_dtype(make_string_dtype(string_encoding_utf_16)).vals();
    EXPECT_EQ(0, memcmp(utf16_string,
                *reinterpret_cast<const char * const *>(x.get_readonly_originptr()),
                sizeof(utf16_string)));
    x = c.as_dtype(make_string_dtype(string_encoding_utf_16)).vals();
    EXPECT_EQ(0, memcmp(utf16_string,
                *reinterpret_cast<const char * const *>(x.get_readonly_originptr()),
                sizeof(utf16_string)));

    // Convert all to utf8 and compare with the reference
    x = a.as_dtype(make_string_dtype(string_encoding_utf_8)).vals();
    EXPECT_EQ(0, memcmp(utf8_string,
                *reinterpret_cast<const char * const *>(x.get_readonly_originptr()),
                sizeof(utf8_string)));
    x = b.as_dtype(make_string_dtype(string_encoding_utf_8)).vals();
    EXPECT_EQ(0, memcmp(utf8_string,
                *reinterpret_cast<const char * const *>(x.get_readonly_originptr()),
                sizeof(utf8_string)));
    x = c.as_dtype(make_string_dtype(string_encoding_utf_8)).vals();
    EXPECT_EQ(0, memcmp(utf8_string,
                *reinterpret_cast<const char * const *>(x.get_readonly_originptr()),
                sizeof(utf8_string)));
}