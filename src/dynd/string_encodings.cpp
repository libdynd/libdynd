//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dynd/type.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/string.hpp>

#include <utf8.h>

using namespace std;
using namespace dynd;

DYND_API int dynd::string_encoding_char_size_table[6] = {
    // string_encoding_ascii
    1,
    // string_encoding_ucs_2
    2,
    // string_encoding_utf_8
    1,
    // string_encoding_utf_16
    2,
    // string_encoding_utf_32
    4,

    // string_encoding_invalid
    0};

namespace {
// The substitute code point when things are invalid
const uint32_t ERROR_SUBSTITUTE_CODEPOINT = (uint32_t)'?';

// The next_* functions advance an iterator pair and return
// the code point that was processed.
static uint32_t next_ascii(const char *&it, const char *DYND_UNUSED(end))
{
  uint32_t result = *reinterpret_cast<const uint8_t *>(it);
  if (result & 0x80) {
    throw string_decode_error(it, it + 1, string_encoding_ascii);
  }
  ++it;
  return result;
}

static uint32_t noerror_next_ascii(const char *&it, const char *DYND_UNUSED(end))
{
  uint32_t result = *reinterpret_cast<const uint8_t *>(it);
  ++it;
  return ((result & 0x80) == 0) ? result : ERROR_SUBSTITUTE_CODEPOINT;
}

static void append_ascii(uint32_t cp, char *&it, char *DYND_UNUSED(end))
{
  if ((cp & ~0x7f) != 0) {
    throw string_encode_error(cp, string_encoding_ascii);
  }
  *it = static_cast<char>(cp);
  ++it;
}

void noerror_append_ascii(uint32_t cp, char *&it, char *DYND_UNUSED(end))
{
  if ((cp & ~0x7f) != 0) {
    cp = ERROR_SUBSTITUTE_CODEPOINT;
  }
  *it = static_cast<char>(cp);
  ++it;
}

uint32_t next_ucs2(const char *&it_raw, const char *DYND_UNUSED(end_raw))
{
  uint32_t cp = *reinterpret_cast<const uint16_t *>(it_raw);
  if (utf8::internal::is_surrogate(cp)) {
    throw string_decode_error(it_raw, it_raw + 2, string_encoding_ucs_2);
  }
  it_raw += 2;
  return cp;
}

uint32_t noerror_next_ucs2(const char *&it_raw, const char *DYND_UNUSED(end_raw))
{
  uint32_t cp = *reinterpret_cast<const uint16_t *>(it_raw);
  it_raw += 2;
  if (utf8::internal::is_surrogate(cp)) {
    return ERROR_SUBSTITUTE_CODEPOINT;
  }
  return cp;
}

void append_ucs2(uint32_t cp, char *&it_raw, char *DYND_UNUSED(end_raw))
{
  uint16_t *&it = reinterpret_cast<uint16_t *&>(it_raw);
  if ((cp & ~0xffff) != 0 || utf8::internal::is_surrogate(cp)) {
    throw string_encode_error(cp, string_encoding_ucs_2);
  }
  *it = static_cast<uint16_t>(cp);
  ++it;
}

void noerror_append_ucs2(uint32_t cp, char *&it_raw, char *DYND_UNUSED(end_raw))
{
  uint16_t *&it = reinterpret_cast<uint16_t *&>(it_raw);
  if ((cp & ~0xffff) != 0 || utf8::internal::is_surrogate(cp)) {
    cp = ERROR_SUBSTITUTE_CODEPOINT;
  }
  *it = static_cast<uint16_t>(cp);
  ++it;
}

uint32_t next_utf8(const char *&it, const char *end)
{
  // const char *saved_it = it;
  uint32_t cp = 0;
  utf8::internal::utf_error err_code = utf8::internal::validate_next(it, end, cp);
  switch (err_code) {
  case utf8::internal::UTF8_OK:
    break;
  case utf8::internal::NOT_ENOUGH_ROOM:
    throw std::runtime_error("Partial UTF8 character at end of buffer");
  case utf8::internal::INVALID_LEAD:
  case utf8::internal::INCOMPLETE_SEQUENCE:
  case utf8::internal::OVERLONG_SEQUENCE:
    // TODO: put the invalid byte range in the exception
    // cout << "invalid sequence: " << string(saved_it, it) << endl;
    throw string_encode_error(cp, string_encoding_utf_8);
  case utf8::internal::INVALID_CODE_POINT:
    throw string_encode_error(cp, string_encoding_utf_8);
  }
  return cp;
}

uint32_t noerror_next_utf8(const char *&it, const char *end)
{
  uint32_t cp = 0;
  // Determine the sequence length based on the lead octet
  std::size_t length = utf8::internal::sequence_length(it);

  // Get trail octets and calculate the code point
  utf8::internal::utf_error err = utf8::internal::UTF8_OK;
  switch (length) {
  case 0:
    return ERROR_SUBSTITUTE_CODEPOINT;
  case 1:
    err = utf8::internal::get_sequence_1(it, end, cp);
    break;
  case 2:
    err = utf8::internal::get_sequence_2(it, end, cp);
    break;
  case 3:
    err = utf8::internal::get_sequence_3(it, end, cp);
    break;
  case 4:
    err = utf8::internal::get_sequence_4(it, end, cp);
    break;
  }

  if (err == utf8::internal::UTF8_OK) {
    // Decoding succeeded. Now, security checks...
    if (utf8::internal::is_code_point_valid(cp)) {
      if (!utf8::internal::is_overlong_sequence(cp, length)) {
        // Passed! Return here.
        ++it;
        return cp;
      } else {
        return ERROR_SUBSTITUTE_CODEPOINT;
      }
    } else {
      return ERROR_SUBSTITUTE_CODEPOINT;
    }
  } else {
    return ERROR_SUBSTITUTE_CODEPOINT;
  }

  return cp;
}

void append_utf8(uint32_t cp, char *&it, char *end)
{
  if (end - it >= 6) {
    it = utf8::append(cp, it);
  } else {
    char tmp[6];
    char *tmp_ptr = tmp;
    tmp_ptr = utf8::append(cp, tmp_ptr);
    if (tmp_ptr - tmp <= end - it) {
      memcpy(it, tmp, tmp_ptr - tmp);
      it += (tmp_ptr - tmp);
    } else {
      throw std::runtime_error("Input too large to convert to destination string");
    }
  }
}

void noerror_append_utf8(uint32_t cp, char *&it, char *end)
{
  if (end - it >= 6) {
    it = utf8::append(cp, it);
  } else {
    char tmp[6];
    char *tmp_ptr = tmp;
    tmp_ptr = utf8::append(cp, tmp_ptr);
    if (tmp_ptr - tmp <= end - it) {
      memcpy(it, tmp, tmp_ptr - tmp);
      it += (tmp_ptr - tmp);
    } else {
      // If it didn't fit, null-terminate
      memset(it, 0, end - it);
      it = end;
    }
  }
}

inline void string_append_utf8(uint32_t cp, std::string &s)
{
  char tmp[6];
  char *tmp_ptr = tmp, *tmp_ptr_end;
  tmp_ptr_end = utf8::append(cp, tmp_ptr);
  while (tmp_ptr < tmp_ptr_end)
    s += *tmp_ptr++;
}

uint32_t next_utf16(const char *&it_raw, const char *end_raw)
{
  uint32_t cp = *reinterpret_cast<const uint16_t *>(it_raw);
  // Take care of surrogate pairs first
  if (utf8::internal::is_lead_surrogate(cp)) {
    if (it_raw + 4 <= end_raw) {
      uint32_t trail_surrogate = *reinterpret_cast<const uint16_t *>(it_raw + 2);
      if (utf8::internal::is_trail_surrogate(trail_surrogate)) {
        cp = (cp << 10) + trail_surrogate + utf8::internal::SURROGATE_OFFSET;
      } else {
        throw string_decode_error(it_raw, it_raw + 4, string_encoding_utf_16);
      }
      it_raw += 2;
    } else {
      throw string_decode_error(it_raw, end_raw, string_encoding_utf_16);
    }

  } else if (utf8::internal::is_trail_surrogate(cp)) {
    // Lone trail surrogate
    throw string_decode_error(it_raw, it_raw + 2, string_encoding_utf_16);
  }
  it_raw += 2;
  return cp;
}

uint32_t noerror_next_utf16(const char *&it_raw, const char *end_raw)
{
  uint32_t cp = *reinterpret_cast<const uint16_t *>(it_raw);
  it_raw += 2;
  // Take care of surrogate pairs first
  if (utf8::internal::is_lead_surrogate(cp)) {
    if (it_raw <= end_raw + 2) {
      uint32_t trail_surrogate = *reinterpret_cast<const uint16_t *>(it_raw);
      it_raw += 2;
      if (utf8::internal::is_trail_surrogate(trail_surrogate)) {
        cp = (cp << 10) + trail_surrogate + utf8::internal::SURROGATE_OFFSET;
      } else {
        return ERROR_SUBSTITUTE_CODEPOINT;
      }
    } else {
      return ERROR_SUBSTITUTE_CODEPOINT;
    }

  } else if (utf8::internal::is_trail_surrogate(cp)) {
    // Lone trail surrogate
    return ERROR_SUBSTITUTE_CODEPOINT;
  }
  return cp;
}

void append_utf16(uint32_t cp, char *&it_raw, char *end_raw)
{
  uint16_t *&it = reinterpret_cast<uint16_t *&>(it_raw);
  uint16_t *end = reinterpret_cast<uint16_t *>(end_raw);
  if (cp > 0xffff) { // make a surrogate pair
    *it = static_cast<uint16_t>((cp >> 10) + utf8::internal::LEAD_OFFSET);
    if (++it >= end) {
      throw std::runtime_error("Input too large to convert to destination string");
    }
    *it = static_cast<uint16_t>((cp & 0x3ff) + utf8::internal::TRAIL_SURROGATE_MIN);
    ++it;
  } else {
    *it = static_cast<uint16_t>(cp);
    ++it;
  }
}

void noerror_append_utf16(uint32_t cp, char *&it_raw, char *end_raw)
{
  uint16_t *&it = reinterpret_cast<uint16_t *&>(it_raw);
  uint16_t *end = reinterpret_cast<uint16_t *>(end_raw);
  if (cp > 0xffff) { // make a surrogate pair
    if (it + 1 < end) {
      *it = static_cast<uint16_t>((cp >> 10) + utf8::internal::LEAD_OFFSET);
      ++it;
      *it = static_cast<uint16_t>((cp & 0x3ff) + utf8::internal::TRAIL_SURROGATE_MIN);
      ++it;
    } else {
      // Null-terminate
      memset(it_raw, 0, end_raw - it_raw);
      it_raw = end_raw;
    }
  } else {
    *it = static_cast<uint16_t>(cp);
    ++it;
  }
}

uint32_t next_utf32(const char *&it_raw, const char *DYND_UNUSED(end_raw))
{
  uint32_t result = *reinterpret_cast<const uint32_t *>(it_raw);
  if (!utf8::internal::is_code_point_valid(result)) {
    throw string_decode_error(it_raw, it_raw + 4, string_encoding_utf_32);
  }
  it_raw += 4;
  return result;
}

uint32_t noerror_next_utf32(const char *&it_raw, const char *DYND_UNUSED(end_raw))
{
  uint32_t result = *reinterpret_cast<const uint32_t *>(it_raw);
  it_raw += 4;
  if (!utf8::internal::is_code_point_valid(result)) {
    return ERROR_SUBSTITUTE_CODEPOINT;
  }
  return result;
}

void append_utf32(uint32_t cp, char *&it_raw, char *DYND_UNUSED(end_raw))
{
  uint32_t *&it = reinterpret_cast<uint32_t *&>(it_raw);
  // uint32_t *end = reinterpret_cast<uint32_t *>(end);
  *it = cp;
  ++it;
}

void noerror_append_utf32(uint32_t cp, char *&it_raw, char *DYND_UNUSED(end_raw))
{
  uint32_t *&it = reinterpret_cast<uint32_t *&>(it_raw);
  // uint32_t *end = reinterpret_cast<uint32_t *>(end);
  *it = cp;
  ++it;
}
} // anonymous namespace

next_unicode_codepoint_t dynd::get_next_unicode_codepoint_function(string_encoding_t encoding,
                                                                   assign_error_mode errmode)
{
  switch (encoding) {
  case string_encoding_ascii:
    return (errmode != assign_error_nocheck) ? next_ascii : noerror_next_ascii;
  case string_encoding_ucs_2:
    return (errmode != assign_error_nocheck) ? next_ucs2 : noerror_next_ucs2;
  case string_encoding_utf_8:
    return (errmode != assign_error_nocheck) ? next_utf8 : noerror_next_utf8;
  case string_encoding_utf_16:
    return (errmode != assign_error_nocheck) ? next_utf16 : noerror_next_utf16;
  case string_encoding_utf_32:
    return (errmode != assign_error_nocheck) ? next_utf32 : noerror_next_utf32;
  default:
    throw runtime_error("get_next_unicode_codepoint_function: Unrecognized string encoding");
  }
}

append_unicode_codepoint_t dynd::get_append_unicode_codepoint_function(string_encoding_t encoding,
                                                                       assign_error_mode errmode)
{
  switch (encoding) {
  case string_encoding_ascii:
    return (errmode != assign_error_nocheck) ? append_ascii : noerror_append_ascii;
  case string_encoding_ucs_2:
    return (errmode != assign_error_nocheck) ? append_ucs2 : noerror_append_ucs2;
  case string_encoding_utf_8:
    return (errmode != assign_error_nocheck) ? append_utf8 : noerror_append_utf8;
  case string_encoding_utf_16:
    return (errmode != assign_error_nocheck) ? append_utf16 : noerror_append_utf16;
  case string_encoding_utf_32:
    return (errmode != assign_error_nocheck) ? append_utf32 : noerror_append_utf32;
  default:
    throw runtime_error("get_append_unicode_codepoint_function: Unrecognized string encoding");
  }
}

template <next_unicode_codepoint_t next_fn>
std::string string_range_as_utf8_string_templ(const char *begin, const char *end)
{
  std::string result;
  uint32_t cp = 0;
  while (begin < end) {
    cp = next_fn(begin, end);
    string_append_utf8(cp, result);
  }
  return result;
}

std::string dynd::string_range_as_utf8_string(string_encoding_t encoding, const char *begin, const char *end,
                                              assign_error_mode errmode)
{
  switch (encoding) {
  case string_encoding_ascii:
  case string_encoding_utf_8:
    // TODO: Validate the input string according to errmode
    return std::string(begin, end);
  case string_encoding_ucs_2:
    if (errmode == assign_error_nocheck) {
      return string_range_as_utf8_string_templ<&noerror_next_ucs2>(begin, end);
    } else {
      return string_range_as_utf8_string_templ<&next_ucs2>(begin, end);
    }
  case string_encoding_utf_16: {
    if (errmode == assign_error_nocheck) {
      return string_range_as_utf8_string_templ<&noerror_next_utf16>(begin, end);
    } else {
      return string_range_as_utf8_string_templ<&next_utf16>(begin, end);
    }
  }
  case string_encoding_utf_32: {
    if (errmode == assign_error_nocheck) {
      return string_range_as_utf8_string_templ<&noerror_next_utf32>(begin, end);
    } else {
      return string_range_as_utf8_string_templ<&next_utf32>(begin, end);
    }
  }
  default: {
    stringstream ss;
    ss << "string_range_as_utf8_string: Unrecognized string encoding";
    ss << encoding;
    throw runtime_error(ss.str());
  }
  }
}

void dynd::print_escaped_unicode_codepoint(std::ostream &o, uint32_t cp, bool single_quote)
{
  if (cp < 0x80) {
    switch (cp) {
    case '\b':
      o << "\\b";
      break;
    case '\f':
      o << "\\f";
      break;
    case '\n':
      o << "\\n";
      break;
    case '\r':
      o << "\\r";
      break;
    case '\t':
      o << "\\t";
      break;
    case '\\':
      o << "\\\\";
      break;
    case '\'':
      o << (single_quote ? "\\'" : "'");
      break;
    case '\"':
      o << (single_quote ? "\"" : "\\\"");
      break;
    default:
      if (cp < 0x20 || cp == 0x7f) {
        o << "\\u";
        hexadecimal_print(o, static_cast<uint16_t>(cp));
      } else {
        o << static_cast<char>(cp);
      }
      break;
    }
  } else if (cp < 0x10000) {
    o << "\\u";
    hexadecimal_print(o, static_cast<uint16_t>(cp));
  } else {
    o << "\\U";
    hexadecimal_print(o, static_cast<uint32_t>(cp));
  }
}

void dynd::print_escaped_utf8_string(std::ostream &o, const char *str_begin, const char *str_end, bool single_quote)
{
  uint32_t cp = 0;

  // Print as an escaped string
  o << (single_quote ? '\'' : '\"');
  while (str_begin < str_end) {
    cp = next_utf8(str_begin, str_end);
    print_escaped_unicode_codepoint(o, cp, single_quote);
  }
  o << (single_quote ? '\'' : '\"');
}

void dynd::append_utf8_codepoint(uint32_t cp, std::string &out_str)
{
  string_append_utf8(cp, out_str);
}

ndt::type dynd::char_type_of_encoding(string_encoding_t encoding)
{
  if (encoding == string_encoding_utf_8) {
    return ndt::make_fixed_bytes(1, 1);
  } else if (encoding == string_encoding_utf_16) {
    return ndt::make_fixed_bytes(2, 2);
  } else {
    return ndt::char_type::make(encoding);
  }
}
