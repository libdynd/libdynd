//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/config.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

enum string_encoding_t {
    string_encoding_ascii,
    string_encoding_ucs_2,
    string_encoding_utf_8,
    string_encoding_utf_16,
    string_encoding_utf_32,

    string_encoding_latin1,
    // TODO: more codepages here

    string_encoding_invalid
};

/**
 * A table of the individual character sizes for
 * the various encodings.
 */
extern DYND_API int string_encoding_char_size_table[6];

/**
 * Returns true if the provided encoding uses a variable-length encoding
 * for each character, for example UTF-8.
 */
inline bool is_variable_length_string_encoding(string_encoding_t encoding)
{
    return encoding == string_encoding_utf_8 || encoding == string_encoding_utf_16;
}

inline std::ostream& operator<<(std::ostream& o, string_encoding_t encoding)
{
    switch (encoding) {
        case string_encoding_ascii:
            o << "ascii";
            break;
        case string_encoding_ucs_2:
            o << "ucs2";
            break;
        case string_encoding_utf_8:
            o << "utf8";
            break;
        case string_encoding_utf_16:
            o << "utf16";
            break;
        case string_encoding_utf_32:
            o << "utf32";
            break;
        case string_encoding_latin1:
            o << "latin1";
            break;
        default:
            o << "unknown string encoding";
            break;
    }

    return o;
}

/**
 * Typedef for getting the next unicode codepoint from a string of a particular
 * encoding.
 *
 * On entry, this function assumes that 'it' and 'end' are appropriately aligned
 * and that (it < end). The variable 'it' is updated in-place to be after the
 * character data representing the returned code point.
 *
 * This function may raise an exception if there is an error.
 */
typedef uint32_t (*next_unicode_codepoint_t)(const char *&it, const char *end);

/**
 * Typedef for appending a unicode codepoint to a string of a particular
 * encoding.
 *
 * On entry, this function assumes that 'it' and 'end' are appropriately aligned
 * and that (it < end). The variable 'it' is updated in-place to be after the
 * character data representing the appended code point.
 *
 * This function may raise an exception if there is an error.
 */
typedef void (*append_unicode_codepoint_t)(uint32_t cp, char *&it, char *end);

DYND_API next_unicode_codepoint_t get_next_unicode_codepoint_function(string_encoding_t encoding, assign_error_mode errmode);
DYND_API append_unicode_codepoint_t get_append_unicode_codepoint_function(string_encoding_t encoding, assign_error_mode errmode);

/**
 * Converts a string buffer provided as a range of bytes into a std::string as UTF8.
 */
DYND_API std::string string_range_as_utf8_string(string_encoding_t encoding, const char *begin, const char *end, assign_error_mode errmode);

/**
 * Prints the given code point to the output stream, escaping it as necessary.
 */
DYND_API void print_escaped_unicode_codepoint(std::ostream &o, uint32_t cp,
                                              bool single_quote);

/**
 * Prints the utf8 string, escaping as necessary.
 */
DYND_API void print_escaped_utf8_string(std::ostream &o, const char *str_begin,
                                        const char *str_end, bool single_quote = false);

/**
 * Prints the utf8 string, escaping as necessary.
 */
inline void print_escaped_utf8_string(std::ostream &o, const std::string &str,
                                      bool single_quote = false)
{
  print_escaped_utf8_string(o, str.data(), str.data() + str.size(),
                            single_quote);
}


DYND_API void append_utf8_codepoint(uint32_t cp, std::string& out_str);

/**
 * Returns the char type corresponding to the encoding. For fixed-sized
 * encodings, this is "char_type[encoding]", and for variable-sized
 * encodings, this is "bytes[1]" or "bytes[2,2]".
 */
DYND_API ndt::type char_type_of_encoding(string_encoding_t encoding);

} // namespace dynd
