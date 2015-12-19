//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <string>
#include <climits>

#include <dynd/parse.hpp>
#include <dynd/callable.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/parse_kernel.hpp>
#include <dynd/functional.hpp>
#include <dynd/types/any_kind_type.hpp>

using namespace std;
using namespace dynd;

// [a-zA-Z_][a-zA-Z0-9_]*
bool dynd::parse_name_no_ws(const char *&rbegin, const char *end, const char *&out_strbegin, const char *&out_strend)
{
  const char *begin = rbegin;
  if (begin == end) {
    return false;
  }
  if (('a' <= *begin && *begin <= 'z') || ('A' <= *begin && *begin <= 'Z') || *begin == '_') {
    ++begin;
  }
  else {
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
                                  const char *&out_strend)
{
  const char *begin = rbegin;
  if (begin == end) {
    return false;
  }
  if (('a' <= *begin && *begin <= 'z') || ('A' <= *begin && *begin <= 'Z')) {
    ++begin;
  }
  else {
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
                                          const char *&out_strend, bool &out_escaped)
{
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
    }
    else if (c == '"') {
      out_strbegin = rbegin + 1;
      out_strend = begin - 1;
      out_escaped = escaped;
      rbegin = begin;
      return true;
    }
  }
}

void dynd::unescape_string(const char *strbegin, const char *strend, std::string &out)
{
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
          }
          else if ('A' <= d && d <= 'F') {
            cp += d - 'A' + 10;
          }
          else if ('a' <= d && d <= 'f') {
            cp += d - 'a' + 10;
          }
          else {
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
          }
          else if ('A' <= d && d <= 'F') {
            cp += d - 'A' + 10;
          }
          else if ('a' <= d && d <= 'f') {
            cp += d - 'a' + 10;
          }
          else {
            cp = '?';
          }
        }
        append_utf8_codepoint(cp, out);
        break;
      }
      default:
        out += '?';
      }
    }
    else {
      out += c;
    }
  }
}

bool dynd::json::parse_number(const char *&rbegin, const char *end, const char *&out_nbegin, const char *&out_nend)
{
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
  }
  else if ('1' <= *begin && *begin <= '9') {
    ++begin;
    while (begin < end && ('0' <= *begin && *begin <= '9')) {
      ++begin;
    }
  }
  else {
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
bool dynd::parse_2digit_int_no_ws(const char *&begin, const char *end, int &out_val)
{
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
bool dynd::parse_1or2digit_int_no_ws(const char *&begin, const char *end, int &out_val)
{
  if (end - begin >= 2) {
    int d0 = begin[0], d1 = begin[1];
    if (d0 >= '0' && d0 <= '9') {
      if (d1 >= '0' && d1 <= '9') {
        begin += 2;
        out_val = (d0 - '0') * 10 + (d1 - '0');
        return true;
      }
      else {
        ++begin;
        out_val = (d0 - '0');
        return true;
      }
    }
  }
  else if (end - begin == 1) {
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
bool dynd::parse_4digit_int_no_ws(const char *&begin, const char *end, int &out_val)
{
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
bool dynd::parse_6digit_int_no_ws(const char *&begin, const char *end, int &out_val)
{
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
static T checked_string_to_signed_int(const char *begin, const char *end)
{
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
  }
  else {
    return negative ? -static_cast<T>(uvalue) : static_cast<T>(uvalue);
  }
}

intptr_t dynd::checked_string_to_intptr(const char *begin, const char *end)
{
  return checked_string_to_signed_int<intptr_t>(begin, end);
}

int64_t dynd::checked_string_to_int64(const char *begin, const char *end)
{
  return checked_string_to_signed_int<int64_t>(begin, end);
}

float dynd::checked_float64_to_float32(double value, assign_error_mode errmode)
{
  union {
    float result;
    char dst[4];
  } out;
  char *src[1] = {reinterpret_cast<char *>(&value)};
  switch (errmode) {
  case assign_error_nocheck:
    dynd::nd::detail::assignment_kernel<float32_type_id, real_kind, float64_type_id, real_kind,
                                        assign_error_nocheck>::single_wrapper(NULL, reinterpret_cast<char *>(&out.dst),
                                                                              src);
    break;
  case assign_error_overflow:
    dynd::nd::detail::assignment_kernel<float32_type_id, real_kind, float64_type_id, real_kind,
                                        assign_error_overflow>::single_wrapper(NULL, reinterpret_cast<char *>(&out.dst),
                                                                               src);
    break;
  case assign_error_fractional:
    dynd::nd::detail::assignment_kernel<float32_type_id, real_kind, float64_type_id, real_kind,
                                        assign_error_fractional>::single_wrapper(NULL,
                                                                                 reinterpret_cast<char *>(&out.dst),
                                                                                 src);
    break;
  case assign_error_inexact:
    dynd::nd::detail::assignment_kernel<float32_type_id, real_kind, float64_type_id, real_kind,
                                        assign_error_inexact>::single_wrapper(NULL, reinterpret_cast<char *>(&out.dst),
                                                                              src);
    break;
  default:
    dynd::nd::detail::assignment_kernel<float32_type_id, real_kind, float64_type_id, real_kind,
                                        assign_error_fractional>::single_wrapper(NULL,
                                                                                 reinterpret_cast<char *>(&out.dst),
                                                                                 src);
    break;
  }
  return out.result;
}

void dynd::string_to_bool(char *out_bool, const char *begin, const char *end, bool option, assign_error_mode errmode)
{
  if (option && parse_na(begin, end)) {
    *out_bool = DYND_BOOL_NA;
    return;
  }
  else {
    size_t size = end - begin;
    if (size == 1) {
      char c = *begin;
      if (c == '0' || c == 'n' || c == 'N' || c == 'f' || c == 'F') {
        *out_bool = 0;
        return;
      }
      else if (errmode == assign_error_nocheck || c == '1' || c == 'y' || c == 'Y' || c == 't' || c == 'T') {
        *out_bool = 1;
        return;
      }
    }
    else if (size == 4) {
      if (errmode == assign_error_nocheck) {
        *out_bool = 1;
        return;
      }
      else if ((begin[0] == 'T' || begin[0] == 't') && (begin[1] == 'R' || begin[1] == 'r') &&
               (begin[2] == 'U' || begin[2] == 'u') && (begin[3] == 'E' || begin[3] == 'e')) {
        *out_bool = 1;
        return;
      }
    }
    else if (size == 5) {
      if ((begin[0] == 'F' || begin[0] == 'f') && (begin[1] == 'A' || begin[1] == 'a') &&
          (begin[2] == 'L' || begin[2] == 'l') && (begin[3] == 'S' || begin[3] == 's') &&
          (begin[4] == 'E' || begin[4] == 'e')) {
        *out_bool = 0;
        return;
      }
      else if (errmode == assign_error_nocheck) {
        *out_bool = 1;
        return;
      }
    }
    else if (size == 0) {
      if (errmode == assign_error_nocheck) {
        *out_bool = 0;
        return;
      }
    }
    else if (size == 2) {
      if ((begin[0] == 'N' || begin[0] == 'n') && (begin[1] == 'O' || begin[1] == 'o')) {
        *out_bool = 0;
        return;
      }
      else if (errmode == assign_error_nocheck ||
               ((begin[0] == 'O' || begin[0] == 'o') && (begin[1] == 'N' || begin[1] == 'n'))) {
        *out_bool = 1;
        return;
      }
    }
    else if (size == 3) {
      if ((begin[0] == 'O' || begin[0] == 'o') && (begin[1] == 'F' || begin[1] == 'f') &&
          (begin[2] == 'F' || begin[2] == 'f')) {
        *out_bool = 0;
        return;
      }
      else if (errmode == assign_error_nocheck ||
               ((begin[0] == 'Y' || begin[0] == 'y') && (begin[1] == 'E' || begin[1] == 'e') &&
                (begin[2] == 'S' || begin[2] == 's'))) {
        *out_bool = 1;
        return;
      }
    }
  }

  stringstream ss;
  ss << "cannot cast string ";
  print_escaped_utf8_string(ss, begin, end);
  if (option) {
    ss << " to ?bool";
  }
  else {
    ss << " to bool";
  }
  throw invalid_argument(ss.str());
}

bool dynd::parse_na(const char *begin, const char *end)
{
  size_t size = end - begin;
  if (size == 0) {
    return true;
  }
  else if (size == 2) {
    if (begin[0] == 'N' && begin[1] == 'A') {
      return true;
    }
  }
  else if (size == 4) {
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

void dynd::parse_uint64(uint64_t &res, const char *begin, const char *end) { res = parse<uint64_t>(begin, end); }

void dynd::parse_int64(int64_t &res, const char *begin, const char *end)
{
  bool negative = false;
  if (begin < end && *begin == '-') {
    negative = true;
    ++begin;
  }

  uint64_t ures;
  parse_uint64(ures, begin, end);

  res = ures;
  if (negative) {
    res = -res;
  }
}

int dynd::parse_double(double &res, const char *begin, const char *end)
{
  try {
    res = parse<double>(begin, end);
  }
  catch (...) {
    return 1;
  }

  return 0;
}

DYND_API struct nd::json::parse nd::json::parse;

DYND_API nd::callable nd::json::parse::make()
{
  std::map<type_id_t, callable> children;
  children[uint8_type_id] = callable::make<parse_kernel<uint8_type_id>>();
  children[uint16_type_id] = callable::make<parse_kernel<uint16_type_id>>();
  children[uint32_type_id] = callable::make<parse_kernel<uint32_type_id>>();
  children[uint64_type_id] = callable::make<parse_kernel<uint64_type_id>>();
  children[int32_type_id] = callable::make<parse_kernel<int32_type_id>>();
  children[option_type_id] = callable::make<parse_kernel<option_type_id>>();
  children[fixed_dim_type_id] = callable::make<parse_kernel<fixed_dim_type_id>>();

  return functional::dispatch(
      ndt::callable_type::make(ndt::make_type<ndt::any_kind_type>(), {ndt::make_type<string>()}),
      [children](const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp)) mutable
      -> callable & { return children[dst_tp.get_type_id()]; });
}
