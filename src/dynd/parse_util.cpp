//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <string>
#include <string>

#include <dynd/config.hpp>
#include <dynd/parse_util.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/type.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/datashape_parser.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

// [a-zA-Z_][a-zA-Z0-9_]*
bool dynd::parse_name_no_ws(const char *&rbegin, const char *end, const char *&out_strbegin, const char *&out_strend) {
  const char *begin = rbegin;
  if (begin == end) {
    return false;
  }
  if (('a' <= *begin && *begin <= 'z') || ('A' <= *begin && *begin <= 'Z') || *begin == '_') {
    ++begin;
  } else {
    return false;
  }
  while (begin < end && (('a' <= *begin && *begin <= 'z') || ('A' <= *begin && *begin <= 'Z') ||
                         ('0' <= *begin && *begin <= '9') || *begin == '_')) {
    ++begin;
  }
  out_strbegin = rbegin;
  out_strend = begin;
  rbegin = begin;
  return true;
}

// [a-zA-Z]+
bool dynd::parse_alpha_name_no_ws(const char *&rbegin, const char *end, const char *&out_strbegin,
                                  const char *&out_strend) {
  const char *begin = rbegin;
  if (begin == end) {
    return false;
  }
  if (('a' <= *begin && *begin <= 'z') || ('A' <= *begin && *begin <= 'Z')) {
    ++begin;
  } else {
    return false;
  }
  while (begin < end && (('a' <= *begin && *begin <= 'z') || ('A' <= *begin && *begin <= 'Z'))) {
    ++begin;
  }
  out_strbegin = rbegin;
  out_strend = begin;
  rbegin = begin;
  return true;
}

bool dynd::parse_doublequote_string_no_ws(const char *&rbegin, const char *end, const char *&out_strbegin,
                                          const char *&out_strend, bool &out_escaped) {
  bool escaped = false;
  const char *begin = rbegin;
  if (!parse_token_no_ws(begin, end, '\"')) {
    return false;
  }
  for (;;) {
    if (begin == end) {
      throw parse_error(rbegin, "string has no ending quote");
    }
    char c = *begin++;
    if (c == '\\') {
      escaped = true;
      if (begin == end) {
        throw parse_error(rbegin, "string has no ending quote");
      }
      c = *begin++;
      switch (c) {
      case '"':
      case '\\':
      case '/':
      case 'b':
      case 'f':
      case 'n':
      case 'r':
      case 't':
        break;
      case 'u': {
        if (end - begin < 4) {
          throw parse_error(begin - 2, "invalid unicode escape sequence in string");
        }
        for (int i = 0; i < 4; ++i) {
          char d = *begin++;
          if (!(('0' <= d && d <= '9') || ('A' <= d && d <= 'F') || ('a' <= d && d <= 'f'))) {
            throw parse_error(begin - 1, "invalid unicode escape sequence in string");
          }
        }
        break;
      }
      case 'U': {
        if (end - begin < 8) {
          throw parse_error(begin - 2, "invalid unicode escape sequence in string");
        }
        for (int i = 0; i < 8; ++i) {
          char d = *begin++;
          if (!(('0' <= d && d <= '9') || ('A' <= d && d <= 'F') || ('a' <= d && d <= 'f'))) {
            throw parse_error(begin - 1, "invalid unicode escape sequence in string");
          }
        }
        break;
      }
      default:
        throw parse_error(begin - 2, "invalid escape sequence in string");
      }
    } else if (c == '"') {
      out_strbegin = rbegin + 1;
      out_strend = begin - 1;
      out_escaped = escaped;
      rbegin = begin;
      return true;
    }
  }
}

void dynd::unescape_string(const char *strbegin, const char *strend, std::string &out) {
  out.resize(0);
  while (strbegin < strend) {
    char c = *strbegin++;
    if (c == '\\') {
      if (strbegin == strend) {
        return;
      }
      c = *strbegin++;
      switch (c) {
      case '"':
      case '\\':
      case '/':
        out += c;
        break;
      case 'b':
        out += '\b';
        break;
      case 'f':
        out += '\f';
        break;
      case 'n':
        out += '\n';
        break;
      case 'r':
        out += '\r';
        break;
      case 't':
        out += '\t';
        break;
      case 'u': {
        if (strend - strbegin < 4) {
          return;
        }
        uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
          char d = *strbegin++;
          cp *= 16;
          if ('0' <= d && d <= '9') {
            cp += d - '0';
          } else if ('A' <= d && d <= 'F') {
            cp += d - 'A' + 10;
          } else if ('a' <= d && d <= 'f') {
            cp += d - 'a' + 10;
          } else {
            cp = '?';
          }
        }
        append_utf8_codepoint(cp, out);
        break;
      }
      case 'U': {
        if (strend - strbegin < 8) {
          return;
        }
        uint32_t cp = 0;
        for (int i = 0; i < 8; ++i) {
          char d = *strbegin++;
          cp *= 16;
          if ('0' <= d && d <= '9') {
            cp += d - '0';
          } else if ('A' <= d && d <= 'F') {
            cp += d - 'A' + 10;
          } else if ('a' <= d && d <= 'f') {
            cp += d - 'a' + 10;
          } else {
            cp = '?';
          }
        }
        append_utf8_codepoint(cp, out);
        break;
      }
      default:
        out += '?';
      }
    } else {
      out += c;
    }
  }
}

bool dynd::json::parse_number(const char *&rbegin, const char *end, const char *&out_nbegin, const char *&out_nend) {
  const char *begin = rbegin;
  if (begin == end) {
    return false;
  }
  // Optional minus sign
  if (*begin == '-') {
    ++begin;
  }
  if (begin == end) {
    return false;
  }
  // Either '0' or a non-zero digit followed by digits
  if (*begin == '0') {
    ++begin;
  } else if ('1' <= *begin && *begin <= '9') {
    ++begin;
    while (begin < end && ('0' <= *begin && *begin <= '9')) {
      ++begin;
    }
  } else {
    return false;
  }
  // Optional decimal point, followed by one or more digits
  if (begin < end && *begin == '.') {
    if (++begin == end) {
      return false;
    }
    if (!('0' <= *begin && *begin <= '9')) {
      return false;
    }
    ++begin;
    while (begin < end && ('0' <= *begin && *begin <= '9')) {
      ++begin;
    }
  }
  // Optional exponent, followed by +/- and some digits
  if (begin < end && (*begin == 'e' || *begin == 'E')) {
    if (++begin == end) {
      return false;
    }
    // +/- is optional
    if (*begin == '+' || *begin == '-') {
      if (++begin == end) {
        return false;
      }
    }
    // At least one digit is required
    if (!('0' <= *begin && *begin <= '9')) {
      return false;
    }
    ++begin;
    while (begin < end && ('0' <= *begin && *begin <= '9')) {
      ++begin;
    }
  }
  out_nbegin = rbegin;
  out_nend = begin;
  rbegin = begin;
  return true;
}

// [0-9][0-9]
bool dynd::parse_2digit_int_no_ws(const char *&begin, const char *end, int &out_val) {
  if (end - begin >= 2) {
    int d0 = begin[0], d1 = begin[1];
    if (d0 >= '0' && d0 <= '9' && d1 >= '0' && d1 <= '9') {
      begin += 2;
      out_val = (d0 - '0') * 10 + (d1 - '0');
      return true;
    }
  }
  return false;
}

// [0-9][0-9]?
bool dynd::parse_1or2digit_int_no_ws(const char *&begin, const char *end, int &out_val) {
  if (end - begin >= 2) {
    int d0 = begin[0], d1 = begin[1];
    if (d0 >= '0' && d0 <= '9') {
      if (d1 >= '0' && d1 <= '9') {
        begin += 2;
        out_val = (d0 - '0') * 10 + (d1 - '0');
        return true;
      } else {
        ++begin;
        out_val = (d0 - '0');
        return true;
      }
    }
  } else if (end - begin == 1) {
    int d0 = begin[0];
    if (d0 >= '0' && d0 <= '9') {
      ++begin;
      out_val = (d0 - '0');
      return true;
    }
  }
  return false;
}

// [0-9][0-9][0-9][0-9]
bool dynd::parse_4digit_int_no_ws(const char *&begin, const char *end, int &out_val) {
  if (end - begin >= 4) {
    int d0 = begin[0], d1 = begin[1], d2 = begin[2], d3 = begin[3];
    if (d0 >= '0' && d0 <= '9' && d1 >= '0' && d1 <= '9' && d2 >= '0' && d2 <= '9' && d3 >= '0' && d3 <= '9') {
      begin += 4;
      out_val = (((d0 - '0') * 10 + (d1 - '0')) * 10 + (d2 - '0')) * 10 + (d3 - '0');
      return true;
    }
  }
  return false;
}

// [0-9][0-9][0-9][0-9][0-9][0-9]
bool dynd::parse_6digit_int_no_ws(const char *&begin, const char *end, int &out_val) {
  if (end - begin >= 6) {
    int d0 = begin[0], d1 = begin[1], d2 = begin[2], d3 = begin[3], d4 = begin[4], d5 = begin[5];
    if (d0 >= '0' && d0 <= '9' && d1 >= '0' && d1 <= '9' && d2 >= '0' && d2 <= '9' && d3 >= '0' && d3 <= '9' &&
        d4 >= '0' && d4 <= '9' && d5 >= '0' && d5 <= '9') {
      begin += 6;
      out_val =
          (((((d0 - '0') * 10 + (d1 - '0')) * 10 + (d2 - '0')) * 10 + (d3 - '0')) * 10 + (d4 - '0')) * 10 + (d5 - '0');
      return true;
    }
  }
  return false;
}

template <class T>
static T checked_string_to_signed_int(const char *begin, const char *end) {
  bool negative = false;
  if (begin < end && *begin == '-') {
    negative = true;
    ++begin;
  }
  uint64_t uvalue = parse<uint64_t>(begin, end);
  if (overflow_check<T>::is_overflow(uvalue, negative)) {
    stringstream ss;
    ss << "overflow converting string ";
    ss.write(begin, end - begin);
    ss << " to " << ndt::make_type<T>();
    throw overflow_error(ss.str());
  } else {
    return negative ? -static_cast<T>(uvalue) : static_cast<T>(uvalue);
  }
}

bool dynd::parse_na(const char *begin, const char *end) {
  size_t size = end - begin;
  if (size == 0) {
    return true;
  } else if (size == 2) {
    if (begin[0] == 'N' && begin[1] == 'A') {
      return true;
    }
  } else if (size == 4) {
    if (((begin[0] == 'N' || begin[0] == 'n') && (begin[1] == 'U' || begin[1] == 'u') &&
         (begin[2] == 'L' || begin[2] == 'l') && (begin[3] == 'L' || begin[3] == 'l'))) {
      return true;
    }
    if (begin[0] == 'N' && begin[1] == 'o' && begin[2] == 'n' && begin[3] == 'e') {
      return true;
    }
  }

  return false;
}

bool json::parse_bool(const char *&begin, const char *&end) {
  bool escaped;
  const char *nbegin;
  const char *nend;

  if (parse_token(begin, end, "true")) {
    return true;
  } else if (parse_token(begin, end, "false")) {
    return false;
  } else if (json::parse_number(begin, end, nbegin, nend)) {
    if (nend - nbegin == 1) {
      if (*nbegin == '0') {
        return false;
      } else if (*nbegin == '1') {
        return true;
      }
    }
  } else if (parse_doublequote_string_no_ws(begin, end, nbegin, nend, escaped)) {
    if (!escaped) {
      return parse<bool>(nbegin, nend);
    } else {
      std::string s;
      unescape_string(nbegin, nend, s);
      return parse<bool>(s.data(), s.data() + s.size());
    }
  }

  std::stringstream ss;
  ss << "cannot cast string ";
  ss.write(begin, end - begin);
  ss << " to bool";
  throw std::invalid_argument(ss.str());
}

bool datashape::parse_quoted_string(const char *&rbegin, const char *end, std::string &out_val) {
  const char *begin = rbegin;
  char beginning_quote = 0;
  out_val = "";
  if (datashape::parse_token(begin, end, '\'')) {
    beginning_quote = '\'';
  } else if (datashape::parse_token(begin, end, '"')) {
    beginning_quote = '"';
  } else {
    return false;
  }
  for (;;) {
    if (begin == end) {
      begin = rbegin;
      skip_whitespace_and_pound_comments(begin, end);
      throw internal_datashape_parse_error(begin, "string has no ending quote");
    }
    char c = *begin++;
    if (c == '\\') {
      if (begin == end) {
        begin = rbegin;
        skip_whitespace_and_pound_comments(begin, end);
        throw internal_datashape_parse_error(begin, "string has no ending quote");
      }
      c = *begin++;
      switch (c) {
      case '"':
      case '\'':
      case '\\':
      case '/':
        out_val += c;
        break;
      case 'b':
        out_val += '\b';
        break;
      case 'f':
        out_val += '\f';
        break;
      case 'n':
        out_val += '\n';
        break;
      case 'r':
        out_val += '\r';
        break;
      case 't':
        out_val += '\t';
        break;
      case 'u': {
        if (end - begin < 4) {
          throw internal_datashape_parse_error(begin - 2, "invalid unicode escape sequence in string");
        }
        uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
          char d = *begin++;
          cp *= 16;
          if ('0' <= d && d <= '9') {
            cp += d - '0';
          } else if ('A' <= d && d <= 'F') {
            cp += d - 'A' + 10;
          } else if ('a' <= d && d <= 'f') {
            cp += d - 'a' + 10;
          } else {
            throw internal_datashape_parse_error(begin - 1, "invalid unicode escape sequence in string");
          }
        }
        append_utf8_codepoint(cp, out_val);
        break;
      }
      default:
        throw internal_datashape_parse_error(begin - 2, "invalid escape sequence in string");
      }
    } else if (c != beginning_quote) {
      out_val += c;
    } else {
      rbegin = begin;
      return true;
    }
  }
}

string_encoding_t datashape::string_to_encoding(const char *error_begin, const std::string &estr) {
  if (estr == "A" || estr == "ascii" || estr == "us-ascii") {
    return string_encoding_ascii;
  } else if (estr == "U8" || estr == "utf8" || estr == "utf-8" || estr == "utf_8") {
    return string_encoding_utf_8;
  } else if (estr == "U16" || estr == "utf16" || estr == "utf-16" || estr == "utf_16") {
    return string_encoding_utf_16;
  } else if (estr == "U32" || estr == "utf32" || estr == "utf-32" || estr == "utf_32") {
    return string_encoding_utf_32;
  } else if (estr == "ucs2" || estr == "ucs-2" || estr == "ucs_2") {
    return string_encoding_ucs_2;
  } else {
    throw internal_datashape_parse_error(error_begin, "unrecognized string encoding");
  }
}
