//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <set>

#include <dynd/types/datashape_parser.hpp>
#include <dynd/parser_util.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/types/array_type.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/types/c_contiguous_type.hpp>
#include <dynd/types/fixed_dim_kind_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/byteswap_type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/ndarrayarg_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/pow_dimsym_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/adapt_type.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/types/kind_sym_type.hpp>
#include <dynd/types/int_kind_sym_type.hpp>
#include <dynd/types/typevar_constructed_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

namespace {
class datashape_parse_error {
  const char *m_position;
  const char *m_message;

public:
  datashape_parse_error(const char *position, const char *message) : m_position(position), m_message(message)
  {
  }
  virtual ~datashape_parse_error()
  {
  }
  const char *get_position() const
  {
    return m_position;
  }
  const char *get_message() const
  {
    return m_message;
  }
};
} // anonymous namespace

// Simple recursive descent parser for a subset of the Blaze datashape grammar.
// (Blaze grammar modified slightly to work this way)

static ndt::type parse_datashape(const char *&begin, const char *end, map<std::string, ndt::type> &symtable);

static const map<std::string, ndt::type> &builtin_types()
{
  static map<std::string, ndt::type> bit;
  if (bit.empty()) {
    bit["void"] = ndt::type::make<void>();
    bit["bool"] = ndt::type::make<bool1>();
    bit["int8"] = ndt::type::make<int8>();
    bit["int16"] = ndt::type::make<int16>();
    bit["int32"] = ndt::type::make<int32>();
    bit["int"] = ndt::type::make<int32>();
    bit["int64"] = ndt::type::make<int64>();
    bit["int128"] = ndt::type::make<int128>();
    bit["intptr"] = ndt::type::make<intptr_t>();
    bit["uint8"] = ndt::type::make<uint8>();
    bit["uint16"] = ndt::type::make<uint16>();
    bit["uint32"] = ndt::type::make<uint32>();
    bit["uint64"] = ndt::type::make<uint64>();
    bit["uint128"] = ndt::type::make<uint128>();
    bit["uintptr"] = ndt::type::make<uintptr_t>();
    bit["float16"] = ndt::type::make<float16>();
    bit["float32"] = ndt::type::make<float32>();
    bit["float64"] = ndt::type::make<float64>();
    bit["real"] = ndt::type::make<double>();
    bit["float128"] = ndt::type::make<float128>();
    bit["complex64"] = ndt::type::make<dynd::complex<float>>();
    bit["complex128"] = ndt::type::make<dynd::complex<double>>();
    bit["complex"] = ndt::type::make<dynd::complex<double>>();
    bit["date"] = ndt::date_type::make();
    bit["time"] = ndt::time_type::make(tz_abstract);
    bit["datetime"] = ndt::datetime_type::make();
    bit["bytes"] = ndt::bytes_type::make(1);
    bit["type"] = ndt::make_type();
    bit["ndarrayarg"] = ndt::make_ndarrayarg();
  }

  return bit;
}

/**
 * Parses a token, skipping whitespace based on datashape's
 * definition of whitespace + comments.
 */
template <int N>
inline bool parse_token_ds(const char *&rbegin, const char *end, const char (&token)[N])
{
  const char *begin = rbegin;
  parse::skip_whitespace_and_pound_comments(begin, end);
  if (parse::parse_token_no_ws(begin, end, token)) {
    rbegin = begin;
    return true;
  } else {
    return false;
  }
}

/**
 * Parses a token, skipping whitespace based on datashape's
 * definition of whitespace + comments.
 */
inline bool parse_token_ds(const char *&rbegin, const char *end, char token)
{
  const char *begin = rbegin;
  parse::skip_whitespace_and_pound_comments(begin, end);
  if (parse::parse_token_no_ws(begin, end, token)) {
    rbegin = begin;
    return true;
  } else {
    return false;
  }
}

static bool parse_name_or_number(const char *&rbegin, const char *end, const char *&out_nbegin, const char *&out_nend)
{
  const char *begin = rbegin;
  // NAME
  if (parse::parse_name_no_ws(begin, end, out_nbegin, out_nend) ||
      parse::parse_unsigned_int_no_ws(begin, end, out_nbegin, out_nend)) {
    rbegin = begin;
    return true;
  }
  return false;
}

static std::string parse_number(const char *&rbegin, const char *end)
{
  const char *begin = rbegin;
  const char *result_begin, *result_end;
  // NUMBER
  parse::skip_whitespace_and_pound_comments(begin, end);
  if (!parse::parse_unsigned_int_no_ws(begin, end, result_begin, result_end)) {
    return std::string();
  }
  rbegin = begin;
  return std::string(result_begin, result_end);
}

static bool parse_quoted_string(const char *&rbegin, const char *end, std::string &out_val)
{
  const char *begin = rbegin;
  char beginning_quote = 0;
  out_val = "";
  if (parse_token_ds(begin, end, '\'')) {
    beginning_quote = '\'';
  } else if (parse_token_ds(begin, end, '"')) {
    beginning_quote = '"';
  } else {
    return false;
  }
  for (;;) {
    if (begin == end) {
      begin = rbegin;
      parse::skip_whitespace_and_pound_comments(begin, end);
      throw datashape_parse_error(begin, "string has no ending quote");
    }
    char c = *begin++;
    if (c == '\\') {
      if (begin == end) {
        begin = rbegin;
        parse::skip_whitespace_and_pound_comments(begin, end);
        throw datashape_parse_error(begin, "string has no ending quote");
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
          throw datashape_parse_error(begin - 2, "invalid unicode escape sequence in string");
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
            throw datashape_parse_error(begin - 1, "invalid unicode escape sequence in string");
          }
        }
        append_utf8_codepoint(cp, out_val);
        break;
      }
      default:
        throw datashape_parse_error(begin - 2, "invalid escape sequence in string");
      }
    } else if (c != beginning_quote) {
      out_val += c;
    } else {
      rbegin = begin;
      return true;
    }
  }
}

// fixed_type : fixed[N] * rhs_expression
static ndt::type parse_fixed_dim_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
    const char *saved_begin = begin;
    std::string dim_size_str = parse_number(begin, end);
    if (dim_size_str.empty()) {
      throw datashape_parse_error(saved_begin, "expected dimension size");
    }
    intptr_t dim_size = (intptr_t)std::atoll(dim_size_str.c_str());
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }
    if (!parse_token_ds(begin, end, '*')) {
      throw datashape_parse_error(begin, "expected dimension separator '*'");
    }
    ndt::type element_tp = parse_datashape(begin, end, symtable);
    if (element_tp.is_null()) {
      throw datashape_parse_error(begin, "expected element type");
    }
    rbegin = begin;
    return ndt::make_fixed_dim(dim_size, element_tp);
  } else {
    throw datashape_parse_error(begin, "expected opening '['");
  }
}

static ndt::type parse_option_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (!parse_token_ds(begin, end, '[')) {
    throw datashape_parse_error(begin, "expected opening '[' after 'option'");
  }
  ndt::type tp = parse_datashape(begin, end, symtable);
  if (tp.is_null()) {
    throw datashape_parse_error(begin, "expected a data type");
  }
  if (!parse_token_ds(begin, end, ']')) {
    throw datashape_parse_error(begin, "expected closing ']'");
  }
  // TODO catch errors, convert them to datashape_parse_error so the position is
  // shown
  rbegin = begin;
  return ndt::option_type::make(tp);
}

static ndt::type parse_adapt_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (!parse_token_ds(begin, end, '[')) {
    throw datashape_parse_error(begin, "expected opening '[' after 'adapt'");
  }
  const char *saved_begin = begin;
  ndt::type proto_tp = parse_datashape(begin, end, symtable);
  if (proto_tp.is_null() || proto_tp.get_type_id() != callable_type_id ||
      proto_tp.extended<ndt::callable_type>()->get_npos() != 1 ||
      proto_tp.extended<ndt::callable_type>()->get_nkwd() != 0) {
    throw datashape_parse_error(saved_begin, "expected a unary function signature");
  }
  if (!parse_token_ds(begin, end, ',')) {
    throw datashape_parse_error(begin, "expected a ,");
  }
  std::string adapt_op;
  ndt::type value_tp;
  if (!parse_quoted_string(begin, end, adapt_op)) {
    throw datashape_parse_error(begin, "expected an an adapt op");
  }
  if (!parse_token_ds(begin, end, ']')) {
    throw datashape_parse_error(begin, "expected closing ']'");
  }
  // TODO catch errors, convert them to datashape_parse_error so the position is
  // shown
  rbegin = begin;
  return ndt::adapt_type::make(proto_tp.extended<ndt::callable_type>()->get_pos_type(0),
                               proto_tp.extended<ndt::callable_type>()->get_return_type(), adapt_op);
}

static string_encoding_t string_to_encoding(const char *error_begin, const std::string &estr)
{
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
    throw datashape_parse_error(error_begin, "unrecognized string encoding");
  }
}

// string_type : string
// This is called after 'string' is already matched
static ndt::type parse_string_parameters(const char *&DYND_UNUSED(rbegin), const char *DYND_UNUSED(end))
{
  return ndt::string_type::make();
}

// fixed_string_type : fixed_string[NUMBER] |
//                     fixed_string[NUMBER,'encoding']
// This is called after 'fixed_string' is already matched
static ndt::type parse_fixed_string_parameters(const char *&rbegin, const char *end)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
    const char *saved_begin = begin;
    std::string value = parse_number(begin, end);
    std::string encoding_str;
    string_encoding_t encoding = string_encoding_utf_8;
    int string_size = 0;
    if (!value.empty()) {
      string_size = atoi(value.c_str());
      if (string_size == 0) {
        throw datashape_parse_error(saved_begin, "string size cannot be zero");
      }
      if (parse_token_ds(begin, end, ',')) {
        saved_begin = begin;
        if (!parse_quoted_string(begin, end, encoding_str)) {
          throw datashape_parse_error(saved_begin, "expected a string encoding");
        }
        encoding = string_to_encoding(saved_begin, encoding_str);
      }
    } else {
      if (!parse_quoted_string(begin, end, encoding_str)) {
        throw datashape_parse_error(saved_begin, "expected a size integer");
      }
      encoding = string_to_encoding(saved_begin, encoding_str);
    }
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::fixed_string_type::make(string_size, encoding);
  }

  throw datashape_parse_error(begin, "expected opening '['");
}

// char_type : char | char[encoding]
// This is called after 'char' is already matched
static ndt::type parse_char_parameters(const char *&rbegin, const char *end)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
    const char *saved_begin = begin;
    std::string encoding_str;
    if (!parse_quoted_string(begin, end, encoding_str)) {
      throw datashape_parse_error(saved_begin, "expected a string encoding");
    }
    string_encoding_t encoding;
    if (!encoding_str.empty()) {
      encoding = string_to_encoding(saved_begin, encoding_str);
    } else {
      throw datashape_parse_error(begin, "expected string encoding");
    }
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::char_type::make(encoding);
  } else {
    return ndt::char_type::make();
  }
}

// complex_type : complex[float_type]
// This is called after 'complex' is already matched
static ndt::type parse_complex_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
    const char *saved_begin = begin;
    ndt::type tp = parse_datashape(begin, end, symtable);
    if (tp.is_null()) {
      throw datashape_parse_error(begin, "expected a type parameter");
    }
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }
    if (tp.get_type_id() == float32_type_id) {
      rbegin = begin;
      return ndt::type::make<dynd::complex<float>>();
    } else if (tp.get_type_id() == float64_type_id) {
      rbegin = begin;
      return ndt::type::make<dynd::complex<double>>();
    } else {
      throw datashape_parse_error(saved_begin, "unsupported real type for complex numbers");
    }
  } else {
    // Default to complex[double] if no parameters are provided
    return ndt::type::make<dynd::complex<double>>();
  }
}

// byteswap_type : byteswap[type]
// This is called after 'byteswap' is already matched
static ndt::type parse_byteswap_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
    ndt::type tp = parse_datashape(begin, end, symtable);
    if (tp.is_null()) {
      throw datashape_parse_error(begin, "expected a type parameter");
    }
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::byteswap_type::make(tp);
  } else {
    throw datashape_parse_error(begin, "expected opening '['");
  }
}

// bytes_type : bytes[align=<alignment>]
// This is called after 'bytes' is already matched
static ndt::type parse_bytes_parameters(const char *&rbegin, const char *end)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
    if (parse_token_ds(begin, end, "align")) {
      // bytes type with an alignment
      if (!parse_token_ds(begin, end, '=')) {
        throw datashape_parse_error(begin, "expected an =");
      }
      std::string align_val = parse_number(begin, end);
      if (align_val.empty()) {
        throw datashape_parse_error(begin, "expected an integer");
      }
      if (!parse_token_ds(begin, end, ']')) {
        throw datashape_parse_error(begin, "expected closing ']'");
      }
      rbegin = begin;
      return ndt::bytes_type::make(atoi(align_val.c_str()));
    }
    throw datashape_parse_error(begin, "expected 'align'");
  } else {
    return ndt::bytes_type::make(1);
  }
}

// fixed_bytes_type : fixed_bytes[<size>] | fixed_bytes[<size>,
// align=<alignment>]
// This is called after 'fixed_bytes' is already matched
static ndt::type parse_fixed_bytes_parameters(const char *&rbegin, const char *end)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
    std::string size_val = parse_number(begin, end);
    if (size_val.empty()) {
      throw datashape_parse_error(begin, "expected 'align' or an integer");
    }
    if (parse_token_ds(begin, end, ']')) {
      // Fixed bytes with just a size parameter
      rbegin = begin;
      return ndt::make_fixed_bytes(atoi(size_val.c_str()), 1);
    }
    if (!parse_token_ds(begin, end, ',')) {
      throw datashape_parse_error(begin, "expected closing ']' or another argument");
    }
    if (!parse_token_ds(begin, end, "align")) {
      throw datashape_parse_error(begin, "expected align= parameter");
    }
    if (!parse_token_ds(begin, end, '=')) {
      throw datashape_parse_error(begin, "expected an =");
    }
    std::string align_val = parse_number(begin, end);
    if (align_val.empty()) {
      throw datashape_parse_error(begin, "expected an integer");
    }
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::make_fixed_bytes(atoi(size_val.c_str()), atoi(align_val.c_str()));
  }
  throw datashape_parse_error(begin, "expected opening '['");
}

// c_contiguous_type : C[child_type]
// This is called after 'C' is already matched
static ndt::type parse_c_contiguous_parameters(const char *&rbegin, const char *end,
                                               map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
    ndt::type tp = parse_datashape(begin, end, symtable);
    if (tp.is_null()) {
      throw datashape_parse_error(begin, "expected a type parameter");
    }
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::c_contiguous_type::make(tp);
  } else {
    throw datashape_parse_error(begin, "expected opening '['");
  }
}

// cuda_host_type : cuda_host[storage_type]
// This is called after 'cuda_host' is already matched
static ndt::type parse_cuda_host_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
#ifdef DYND_CUDA
    ndt::type tp = parse_datashape(begin, end, symtable);
    if (tp.is_null()) {
      throw datashape_parse_error(begin, "expected a type parameter");
    }
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::make_cuda_host(tp);
#else
    // Silence the unused parameter warning
    symtable.empty();
    throw datashape_parse_error(begin, "cuda_host type is not available");
#endif // DYND_CUDA
  } else {
    throw datashape_parse_error(begin, "expected opening '['");
  }
}

// cuda_device_type : cuda_device[storage_type]
// This is called after 'cuda_device' is already matched
static ndt::type parse_cuda_device_parameters(const char *&rbegin, const char *end,
                                              map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
#ifdef DYND_CUDA
    ndt::type tp = parse_datashape(begin, end, symtable);
    if (tp.is_null()) {
      throw datashape_parse_error(begin, "expected a type parameter");
    }
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::make_cuda_device(tp);
#else
    // Silence the unused parameter warning
    symtable.empty();
    throw datashape_parse_error(begin, "cuda_device type is not available");
#endif // DYND_CUDA
  } else {
    throw datashape_parse_error(begin, "expected opening '['");
  }
}

// datetime_type : datetime[tz='timezone']
// This is called after 'datetime' is already matched
static ndt::type parse_datetime_parameters(const char *&rbegin, const char *end)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
    datetime_tz_t timezone = tz_abstract;
    std::string unit_str;
    const char *saved_begin = begin;
    // Parse the timezone
    if (!parse_token_ds(begin, end, "tz")) {
      throw datashape_parse_error(begin, "expected tz= parameter");
    }
    if (!parse_token_ds(begin, end, '=')) {
      throw datashape_parse_error(begin, "expected '='");
    }
    std::string timezone_str;
    saved_begin = begin;
    if (!parse_quoted_string(begin, end, timezone_str)) {
      throw datashape_parse_error(begin, "expected a time zone string");
    }
    if (timezone_str == "abstract") {
      timezone = tz_abstract;
    } else if (timezone_str == "UTC") {
      timezone = tz_utc;
    } else {
      throw datashape_parse_error(saved_begin, "invalid time zone");
    }
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }

    rbegin = begin;
    return ndt::datetime_type::make(timezone);
  } else {
    return ndt::datetime_type::make();
  }
}

// time_type : time[tz='timezone']
// This is called after 'datetime' is already matched
static ndt::type parse_time_parameters(const char *&rbegin, const char *end)
{
  const char *begin = rbegin;
  if (parse_token_ds(begin, end, '[')) {
    datetime_tz_t timezone = tz_abstract;
    std::string unit_str;
    const char *saved_begin = begin;
    // Parse the timezone
    if (!parse_token_ds(begin, end, "tz")) {
      throw datashape_parse_error(begin, "expected tz= parameter");
    }
    if (!parse_token_ds(begin, end, '=')) {
      throw datashape_parse_error(begin, "expected '='");
    }
    std::string timezone_str;
    saved_begin = begin;
    if (!parse_quoted_string(begin, end, timezone_str)) {
      throw datashape_parse_error(begin, "expected a time zone string");
    }
    if (timezone_str == "abstract") {
      timezone = tz_abstract;
    } else if (timezone_str == "UTC") {
      timezone = tz_utc;
    } else {
      throw datashape_parse_error(saved_begin, "invalid time zone");
    }
    if (!parse_token_ds(begin, end, ']')) {
      throw datashape_parse_error(begin, "expected closing ']'");
    }

    rbegin = begin;
    return ndt::time_type::make(timezone);
  } else {
    return ndt::time_type::make(tz_abstract);
  }
}

static ndt::type parse_unaligned_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (!parse_token_ds(begin, end, '[')) {
    throw datashape_parse_error(begin, "expected opening '[' after 'unaligned'");
  }
  ndt::type tp = parse_datashape(begin, end, symtable);
  if (tp.is_null()) {
    throw datashape_parse_error(begin, "expected a data type");
  }
  if (!parse_token_ds(begin, end, ']')) {
    throw datashape_parse_error(begin, "expected closing ']'");
  }
  // TODO catch errors, convert them to datashape_parse_error so the position is
  // shown
  rbegin = begin;
  return ndt::make_unaligned(tp);
}

static ndt::type parse_pointer_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (!parse_token_ds(begin, end, '[')) {
    throw datashape_parse_error(begin, "expected opening '[' after 'pointer'");
  }
  ndt::type tp = parse_datashape(begin, end, symtable);
  if (tp.is_null()) {
    throw datashape_parse_error(begin, "expected a data type");
  }
  if (!parse_token_ds(begin, end, ']')) {
    throw datashape_parse_error(begin, "expected closing ']'");
  }
  // TODO catch errors, convert them to datashape_parse_error so the position is
  // shown
  rbegin = begin;
  return ndt::pointer_type::make(tp);
}

static ndt::type parse_array_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  if (!parse_token_ds(begin, end, '[')) {
    throw datashape_parse_error(begin, "expected opening '[' after 'ref'");
  }
  ndt::type tp = parse_datashape(begin, end, symtable);
  if (tp.is_null()) {
    throw datashape_parse_error(begin, "expected a data type");
  }
  if (!parse_token_ds(begin, end, ']')) {
    throw datashape_parse_error(begin, "expected closing ']'");
  }
  // TODO catch errors, convert them to datashape_parse_error so the position is
  // shown
  rbegin = begin;
  return ndt::array_type::make(tp);
}

// datashape_list : datashape COMMA datashape_list RBRACKET
//                | datashape RBRACKET
static nd::array parse_datashape_list(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;

  vector<ndt::type> dlist;
  ndt::type arg = parse_datashape(begin, end, symtable);
  if (arg.is_null()) {
    return nd::array();
  }
  dlist.push_back(arg);
  for (;;) {
    if (!parse_token_ds(begin, end, ',')) {
      if (!parse_token_ds(begin, end, ']')) {
        return nd::array();
      } else {
        break;
      }
    }
    arg = parse_datashape(begin, end, symtable);
    if (arg.is_null()) {
      throw datashape_parse_error(begin, "Expected a dynd type or a terminating ']'");
    }
    dlist.push_back(arg);
  }

  rbegin = begin;
  return dlist;
}

// integer_list : INTEGER COMMA integer_list RBRACKET
//              | INTEGER RBRACKET
static nd::array parse_integer_list(const char *&rbegin, const char *end)
{
  const char *begin = rbegin;

  vector<int64_t> dlist;
  const char *strbegin, *strend;
  if (!parse::parse_int_no_ws(begin, end, strbegin, strend)) {
    return nd::array();
  }
  dlist.push_back(parse::checked_string_to_int64(strbegin, strend));
  for (;;) {
    if (!parse_token_ds(begin, end, ',')) {
      if (parse_token_ds(begin, end, ']')) {
        rbegin = begin;
        return dlist;
      } else {
        return nd::array();
      }
    }
    parse::skip_whitespace_and_pound_comments(begin, end);
    if (!parse::parse_int_no_ws(begin, end, strbegin, strend)) {
      throw datashape_parse_error(begin, "Expected an integer or a terminating ']'");
    }
    dlist.push_back(parse::checked_string_to_int64(strbegin, strend));
  }
}

// string_list : STRING COMMA string_list RBRACKET
//             | STRING RBRACKET
static nd::array parse_string_list(const char *&rbegin, const char *end)
{
  const char *begin = rbegin;

  vector<std::string> dlist;
  std::string str;
  if (!parse_quoted_string(begin, end, str)) {
    return nd::array();
  }
  dlist.push_back(str);
  for (;;) {
    if (!parse_token_ds(begin, end, ',')) {
      if (!parse_token_ds(begin, end, ']')) {
        return nd::array();
      } else {
        break;
      }
    }
    if (!parse_quoted_string(begin, end, str)) {
      throw datashape_parse_error(begin, "Expected a string");
    }
    dlist.push_back(str);
  }

  rbegin = begin;
  return dlist;
}

// list_type_arg : LBRACKET RBRACKET
//               | LBRACKET datashape_list
//               | LBRACKET integer_list
//               | LBRACKET string_list
// type_arg : datashape
//          | INTEGER
//          | STRING
//          | list_type_arg
static nd::array parse_type_arg(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;

  parse::skip_whitespace_and_pound_comments(begin, end);

  const char *strbegin, *strend;
  if (parse::parse_int_no_ws(begin, end, strbegin, strend)) {
    rbegin = begin;
    return parse::checked_string_to_int64(strbegin, strend);
  }

  std::string str;
  if (parse_quoted_string(begin, end, str)) {
    rbegin = begin;
    return str;
  }

  if (parse::parse_token(begin, end, '[')) {
    nd::array result;
    result = parse_integer_list(begin, end);
    if (result.is_null()) {
      result = parse_string_list(begin, end);
    }
    if (result.is_null()) {
      result = parse_datashape_list(begin, end, symtable);
    }
    if (result.is_null()) {
      result = nd::empty(0, ndt::type::make<void>());
    }
    rbegin = begin;
    return result;
  }

  ndt::type tp = parse_datashape(begin, end, symtable);
  if (!tp.is_null()) {
    rbegin = begin;
    return tp;
  }

  return nd::array();
}

// type_arg_list : type_arg COMMA type_arg_list
//               | type_kwarg_list
//               | type_arg
// type_kwarg_list : type_kwarg COMMA type_kwarg_list
//                 | type_kwarg
// type_kwarg : NAME_LOWER EQUAL type_arg
// type_constr_args : LBRACKET type_arg_list RBRACKET
nd::array dynd::parse_type_constr_args(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  nd::array result;

  const char *begin = rbegin;
  if (!parse_token_ds(begin, end, '[')) {
    // Return an empty array if there is no leading bracket
    return result;
  }

  if (parse_token_ds(begin, end, ']')) {
    return nd::empty(ndt::tuple_type::make({ndt::tuple_type::make(), ndt::struct_type::make()}));
  }

  vector<nd::array> pos_args;
  vector<nd::array> kw_args;
  vector<std::string> kw_names;

  const char *field_name_begin, *field_name_end;
  bool done = false;

  // First parse all the positional arguments
  for (;;) {
    parse::skip_whitespace_and_pound_comments(begin, end);
    // Look ahead to see the ']' or whether there's a keyword argument
    const char *saved_begin = begin;
    if (parse::parse_name_no_ws(begin, end, field_name_begin, field_name_end) && parse_token_ds(begin, end, ':')) {
      begin = saved_begin;
      break;
    }
    begin = saved_begin;
    // Parse one positional argument
    nd::array arg = parse_type_arg(begin, end, symtable);
    if (arg.is_null()) {
      throw datashape_parse_error(saved_begin, "Expected a positional or keyword type argument");
    }
    pos_args.push_back(arg);
    if (!parse_token_ds(begin, end, ',')) {
      if (parse_token_ds(begin, end, ']')) {
        done = true;
      }
      break;
    }
  }

  // Now parse all the keyword arguments
  if (!done) {
    for (;;) {
      const char *saved_begin = begin;
      parse::skip_whitespace_and_pound_comments(begin, end);
      if (!parse::parse_name_no_ws(begin, end, field_name_begin, field_name_end)) {
        throw datashape_parse_error(saved_begin, "Expected a keyword name or terminating ']'");
      }
      if (!parse_token_ds(begin, end, ':')) {
        throw datashape_parse_error(begin, "Expected ':' between keyword name and parameter");
      }
      nd::array arg = parse_type_arg(begin, end, symtable);
      if (arg.is_null()) {
        throw datashape_parse_error(begin, "Expected keyword argument value");
      }
      kw_args.push_back(arg);
      kw_names.push_back(std::string(field_name_begin, field_name_end));
      if (!parse_token_ds(begin, end, ',')) {
        if (!parse_token_ds(begin, end, ']')) {
          throw datashape_parse_error(begin, "Expected a ',' or ']'");
        } else {
          break;
        }
      }
    }
  }

  // Create type "((type0, ...), {kw0: kwtype0, ...})"
  vector<ndt::type> pos_arg_types;
  transform(pos_args.begin(), pos_args.end(), back_inserter(pos_arg_types),
            [](const nd::array &a) { return a.get_type(); });

  vector<ndt::type> kw_arg_types;
  transform(kw_args.begin(), kw_args.end(), back_inserter(kw_arg_types),
            [](const nd::array &a) { return a.get_type(); });

  ndt::type result_tp =
      ndt::tuple_type::make({ndt::tuple_type::make(pos_arg_types), ndt::struct_type::make(kw_names, kw_arg_types)});

  result = nd::empty(result_tp);
  nd::array pos = result(0);
  for (size_t i = 0; i != pos_args.size(); ++i) {
    pos.vals_at(i) = pos_args[i];
  }
  nd::array kw = result(1);
  for (size_t i = 0; i != kw_args.size(); ++i) {
    kw.vals_at(i) = kw_args[i];
  }

  rbegin = begin;
  return result;
}

// record_item_bare : BARENAME COLON rhs_expression
static bool parse_struct_item_bare(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable,
                                   std::string &out_field_name, ndt::type &out_field_type)
{
  const char *begin = rbegin;
  const char *field_name_begin, *field_name_end;
  parse::skip_whitespace_and_pound_comments(begin, end);
  if (parse::parse_name_no_ws(begin, end, field_name_begin, field_name_end)) {
    // We successfully parsed a name with no whitespace
    // We don't need to do anything else, because field_name_begin
  } else {
    // This struct item cannot be parsed. Ergo, we return false for failure.
    return false;
  }
  if (!parse_token_ds(begin, end, ':')) {
    throw datashape_parse_error(begin, "expected ':' after record item name");
  }
  out_field_type = parse_datashape(begin, end, symtable);
  if (out_field_type.is_null()) {
    throw datashape_parse_error(begin, "expected a data type");
  }

  out_field_name.assign(field_name_begin, field_name_end);
  rbegin = begin;
  return true;
}

// struct_item_general : struct_item_bare |
//                       QUOTEDNAME COLON rhs_expression
static bool parse_struct_item_general(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable,
                                      std::string &out_field_name, ndt::type &out_field_type)
{
  const char *begin = rbegin;
  const char *field_name_begin, *field_name_end;
  // quoted_out_val and quoted_name are used to hold the field name and to
  // denote if the data given
  //  to this function needed special handling due to quoting of the struct
  //  field names.
  std::string quoted_out_val;
  bool quoted_name = false;
  parse::skip_whitespace_and_pound_comments(begin, end);
  if (parse::parse_name_no_ws(begin, end, field_name_begin, field_name_end)) {
    // We successfully parsed a name with no whitespace
    // We don't need to do anything else, because field_name_begin
  } else if (parse_quoted_string(begin, end, quoted_out_val)) {
    // parse_quoted_string must return a new string for us to use because it
    // will parse
    //  and potentially replace things in the string (like escaped characters)
    // It will also remove the surrounding quotes.
    quoted_name = true;
  } else {
    // This struct item cannot be parsed. Ergo, we return false for failure.
    return false;
  }
  if (!parse_token_ds(begin, end, ':')) {
    throw datashape_parse_error(begin, "expected ':' after record item name");
  }
  out_field_type = parse_datashape(begin, end, symtable);
  if (out_field_type.is_null()) {
    throw datashape_parse_error(begin, "expected a data type");
  }

  if (!quoted_name) {
    // A name that isn't quoted is probably the common case
    out_field_name.assign(field_name_begin, field_name_end);
  } else {
    // If a field name was quoted, parse_quoted_string() will have parsed and
    // un/re-escaped everything and returned a new string
    // The Return of the String is why we have two different
    // out_field_name.assign() cases
    out_field_name.assign(quoted_out_val);
  }
  rbegin = begin;
  return true;
}

// struct : LBRACE record_item record_item* RBRACE
static ndt::type parse_struct(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  vector<std::string> field_name_list;
  vector<ndt::type> field_type_list;
  std::string field_name;
  ndt::type field_type;
  bool variadic = false;

  if (!parse_token_ds(begin, end, '{')) {
    return ndt::type(uninitialized_type_id);
  }
  if (parse_token_ds(begin, end, '}')) {
    // Empty struct
    rbegin = begin;
    return ndt::struct_type::make();
  }
  for (;;) {
    if (parse_token_ds(begin, end, "...")) {
      if (!parse_token_ds(begin, end, '}')) {
        throw datashape_parse_error(begin, "expected '}'");
      }
      variadic = true;
      break;
    }

    const char *saved_begin = begin;
    parse::skip_whitespace_and_pound_comments(begin, end);
    if (parse_struct_item_general(begin, end, symtable, field_name, field_type)) {
      field_name_list.push_back(field_name);
      field_type_list.push_back(field_type);
    } else {
      throw datashape_parse_error(saved_begin, "expected a record item");
    }

    if (parse_token_ds(begin, end, ',')) {
      if (!field_name_list.empty() && parse_token_ds(begin, end, '}')) {
        break;
      }
    } else if (parse_token_ds(begin, end, '}')) {
      break;
    } else {
      throw datashape_parse_error(begin, "expected ',' or '}'");
    }
  }

  rbegin = begin;
  return ndt::struct_type::make(field_name_list, field_type_list, variadic);
}

// funcproto_kwds : record_item, record_item*
static ndt::type parse_funcproto_kwds(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  vector<std::string> field_name_list;
  vector<ndt::type> field_type_list;
  std::string field_name;
  ndt::type field_type;
  bool variadic = false;

  for (;;) {
    // Check for variadic ending
    if (parse_token_ds(begin, end, "...")) {
      if (!parse_token_ds(begin, end, ')')) {
        throw datashape_parse_error(begin, "expected ',' or ')' in callable prototype");
      }
      variadic = true;
      break;
    }

    const char *saved_begin = begin;
    parse::skip_whitespace_and_pound_comments(begin, end);
    if (parse_struct_item_bare(begin, end, symtable, field_name, field_type)) {
      field_name_list.push_back(field_name);
      field_type_list.push_back(field_type);
    } else {
      throw datashape_parse_error(saved_begin, "expected a kwd arg in callable prototype");
    }

    if (parse_token_ds(begin, end, ',')) {
      if (!field_name_list.empty() && parse_token_ds(begin, end, ')')) {
        break;
      }
    } else if (parse_token_ds(begin, end, ')')) {
      break;
    } else {
      throw datashape_parse_error(begin, "expected ',' or ')' in callable prototype");
    }
  }

  rbegin = begin;
  return ndt::struct_type::make(field_name_list, field_type_list, variadic);
}

// tuple : LPAREN tuple_item tuple_item* RPAREN
// funcproto : tuple -> type
static ndt::type parse_tuple_or_funcproto(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  vector<ndt::type> field_type_list;
  bool variadic = false;

  if (!parse_token_ds(begin, end, '(')) {
    return ndt::type(uninitialized_type_id);
  }
  if (!parse_token_ds(begin, end, ')')) {
    for (;;) {
      ndt::type tp;
      // Look ahead to see if we've got "BARENAME:" or "..., BARENAME:" coming
      // next, and if so, parse the keyword arguments and return.
      const char *saved_begin = begin, *kwds_begin = begin;
      const char *field_name_begin, *field_name_end;
      if (parse_token_ds(begin, end, "...") && parse_token_ds(begin, end, ',')) {
        variadic = true;
        kwds_begin = begin;
      }
      parse::skip_whitespace_and_pound_comments(begin, end);
      if (parse::parse_name_no_ws(begin, end, field_name_begin, field_name_end)) {
        if (parse_token_ds(begin, end, ':')) {
          // process the keyword arguments
          ndt::type funcproto_kwd;
          begin = kwds_begin;
          funcproto_kwd = parse_funcproto_kwds(begin, end, symtable);
          if (!funcproto_kwd.is_null()) {
            if (!parse_token_ds(begin, end, "->")) {
              rbegin = begin;
              return ndt::tuple_type::make(field_type_list);
            }

            ndt::type return_type = parse_datashape(begin, end, symtable);
            if (return_type.is_null()) {
              throw datashape_parse_error(begin, "expected function prototype return type");
            }
            rbegin = begin;
            return ndt::callable_type::make(return_type, ndt::tuple_type::make(field_type_list, variadic),
                                            funcproto_kwd);
          } else {
            throw datashape_parse_error(begin, "expected funcproto keyword arguments");
          }
        }
      }
      begin = saved_begin;

      // Look ahead again to see if the tuple ends with "...)", in which case
      // it's a variadic tuple.
      if (parse_token_ds(begin, end, "...")) {
        if (parse_token_ds(begin, end, ')')) {
          variadic = true;
          break;
        }
      }
      begin = saved_begin;

      tp = parse_datashape(begin, end, symtable);

      if (tp.get_type_id() != uninitialized_type_id) {
        field_type_list.push_back(tp);
      } else {
        throw datashape_parse_error(begin, "expected a type");
      }

      if (parse_token_ds(begin, end, ',')) {
        if (!field_type_list.empty() && parse_token_ds(begin, end, ')')) {
          break;
        }
      } else if (parse_token_ds(begin, end, ')')) {
        break;
      } else {
        throw datashape_parse_error(begin, "expected ',' or ')'");
      }
    }
  }

  // It might be a function prototype, check for the "->" token
  if (!parse_token_ds(begin, end, "->")) {
    rbegin = begin;
    return ndt::tuple_type::make(field_type_list, variadic);
  }

  ndt::type return_type = parse_datashape(begin, end, symtable);
  if (return_type.is_null()) {
    throw datashape_parse_error(begin, "expected function prototype return type");
  }
  rbegin = begin;
  // TODO: I suspect because of the change away from immutable default
  // construction, and
  //       the requirement that arrays into callable constructors are
  //       immutable, that too
  //       many copies may be occurring.
  return ndt::callable_type::make(return_type, ndt::tuple_type::make(field_type_list, variadic));
}

//    datashape_nooption : dim ASTERISK datashape
//                       | dtype
static ndt::type parse_datashape_nooption(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  ndt::type result;
  const char *begin = rbegin;
  parse::skip_whitespace_and_pound_comments(begin, end);
  // First try "dim ASTERISK ASTERISK datashape", then "dim ASTERISK datashape"
  const char *nbegin, *nend;
  if (parse_name_or_number(begin, end, nbegin, nend)) {
    if (parse_token_ds(begin, end, "**")) {
      // Save the extents of the base dim token
      const char *bbegin = nbegin;
      const char *bend = nend;
      if (parse_name_or_number(begin, end, nbegin, nend)) {
        if ('1' <= *nbegin && *nbegin <= '9') {
          intptr_t exponent = parse::checked_string_to_intptr(nbegin, nend);
          if (!parse_token_ds(begin, end, '*')) {
            throw datashape_parse_error(begin, "expected a '*' after dimensional power");
          }
          ndt::type element_tp = parse_datashape(begin, end, symtable);
          if ('0' <= *bbegin && *bbegin <= '9') {
            intptr_t dim_size = parse::checked_string_to_intptr(bbegin, bend);
            result = ndt::make_fixed_dim(dim_size, element_tp, exponent);
          } else if (parse::compare_range_to_literal(bbegin, bend, "var")) {
            result = ndt::var_dim_type::make(element_tp, exponent);
          } else if (parse::compare_range_to_literal(bbegin, bend, "Fixed")) {
            result = make_fixed_dim_kind(element_tp, exponent);
          } else if (isupper(*bbegin)) {
            result = ndt::typevar_dim_type::make(std::string(bbegin, bend), element_tp, exponent);
          } else {
            throw datashape_parse_error(bbegin, "invalid dimension type for base of dimensional power");
          }
        } else if (isupper(*nbegin)) {
          std::string exponent_name(nbegin, nend);
          if (parse_token_ds(begin, end, '*')) {
            if ('0' <= *bbegin && *bbegin <= '9') {
              intptr_t dim_size = parse::checked_string_to_intptr(bbegin, bend);
              result = ndt::make_pow_dimsym(ndt::make_fixed_dim(dim_size, ndt::type::make<void>()), exponent_name,
                                            parse_datashape(begin, end, symtable));
            } else if (parse::compare_range_to_literal(bbegin, bend, "var")) {
              result = ndt::make_pow_dimsym(ndt::var_dim_type::make(ndt::type::make<void>()), exponent_name,
                                            parse_datashape(begin, end, symtable));
            } else if (parse::compare_range_to_literal(bbegin, bend, "Fixed")) {
              result = ndt::make_pow_dimsym(ndt::make_fixed_dim_kind(ndt::type::make<void>()), exponent_name,
                                            parse_datashape(begin, end, symtable));
            } else if (isupper(*bbegin)) {
              result =
                  ndt::make_pow_dimsym(ndt::typevar_dim_type::make(std::string(bbegin, bend), ndt::type::make<void>()),
                                       exponent_name, parse_datashape(begin, end, symtable));
            } else {
              throw datashape_parse_error(bbegin, "invalid dimension type for base of dimensional power");
            }
          }
        } else {
          throw datashape_parse_error(begin, "expected a number or a typevar symbol");
        }
      } else {
        throw datashape_parse_error(begin, "expected a number or a typevar symbol");
      }
    } else if (parse_token_ds(begin, end, '*')) {
      ndt::type element_tp = parse_datashape(begin, end, symtable);
      if (element_tp.is_null()) {
        throw datashape_parse_error(begin, "expected a dynd type");
      }
      // No type constructor args, just a dim type
      if ('0' <= *nbegin && *nbegin <= '9') {
        intptr_t size = parse::checked_string_to_intptr(nbegin, nend);
        result = ndt::make_fixed_dim(size, element_tp);
      } else if (parse::compare_range_to_literal(nbegin, nend, "var")) {
        result = ndt::var_dim_type::make(element_tp);
      } else if (parse::compare_range_to_literal(nbegin, nend, "Fixed")) {
        result = ndt::make_fixed_dim_kind(element_tp);
      } else if (isupper(*nbegin)) {
        result = ndt::typevar_dim_type::make(std::string(nbegin, nend), element_tp);
      } else {
        parse::skip_whitespace_and_pound_comments(rbegin, end);
        throw datashape_parse_error(rbegin, "unrecognized dimension type");
      }
    } else if (parse_token_ds(begin, end, "...")) { // ELLIPSIS
      // A named ellipsis dim
      if (parse_token_ds(begin, end, '*')) { // ASTERISK
        // An unnamed ellipsis dim
        ndt::type element_tp = parse_datashape(begin, end, symtable);
        if (element_tp.is_null()) {
          throw datashape_parse_error(begin, "expected a dynd type");
        }
        result = ndt::make_ellipsis_dim(std::string(nbegin, nend), element_tp);
      } else {
        throw datashape_parse_error(begin, "expected a '*'");
      }
    } else if (parse_token_ds(begin, end, '|')) { // TYPE
      result = ndt::make_type(parse_datashape(begin, end, symtable));
    } else if (parse::compare_range_to_literal(nbegin, nend, "string")) {
      result = parse_string_parameters(begin, end);
    } else if (parse::compare_range_to_literal(nbegin, nend, "fixed_string")) {
      result = parse_fixed_string_parameters(begin, end);
    } else if (parse::compare_range_to_literal(nbegin, nend, "complex")) {
      result = parse_complex_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "datetime")) {
      result = parse_datetime_parameters(begin, end);
    } else if (parse::compare_range_to_literal(nbegin, nend, "time")) {
      result = parse_time_parameters(begin, end);
    } else if (parse::compare_range_to_literal(nbegin, nend, "unaligned")) {
      result = parse_unaligned_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "pointer")) {
      result = parse_pointer_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "array")) {
      result = parse_array_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "char")) {
      result = parse_char_parameters(begin, end);
    } else if (parse::compare_range_to_literal(nbegin, nend, "byteswap")) {
      result = parse_byteswap_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "bytes")) {
      result = parse_bytes_parameters(begin, end);
    } else if (parse::compare_range_to_literal(nbegin, nend, "fixed_bytes")) {
      result = parse_fixed_bytes_parameters(begin, end);
    } else if (parse::compare_range_to_literal(nbegin, nend, "C")) {
      result = parse_c_contiguous_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "cuda_host")) {
      result = parse_cuda_host_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "cuda_device")) {
      result = parse_cuda_device_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "fixed")) {
      result = parse_fixed_dim_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "option")) {
      result = parse_option_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "adapt")) {
      result = parse_adapt_parameters(begin, end, symtable);
    } else if (parse::compare_range_to_literal(nbegin, nend, "Any")) {
      result = ndt::any_kind_type::make();
    } else if (parse::compare_range_to_literal(nbegin, nend, "Scalar")) {
      result = ndt::scalar_kind_type::make();
    } else if (parse::compare_range_to_literal(nbegin, nend, "Categorical")) {
      result = ndt::categorical_kind_type::make();
    } else if (parse::compare_range_to_literal(nbegin, nend, "FixedBytes")) {
      result = ndt::fixed_bytes_kind_type::make();
    } else if (parse::compare_range_to_literal(nbegin, nend, "FixedString")) {
      result = ndt::fixed_string_kind_type::make();
    } else if (parse::compare_range_to_literal(nbegin, nend, "Bool")) {
      result = ndt::make_kind_sym(bool_kind);
    } else if (parse::compare_range_to_literal(nbegin, nend, "Int")) {
      result = ndt::make_int_kind_sym();
    } else if (parse::compare_range_to_literal(nbegin, nend, "UInt")) {
      result = ndt::make_kind_sym(uint_kind);
    } else if (parse::compare_range_to_literal(nbegin, nend, "SInt")) {
      result = ndt::make_kind_sym(sint_kind);
    } else if (parse::compare_range_to_literal(nbegin, nend, "Real")) {
      result = ndt::make_kind_sym(real_kind);
    } else if (parse::compare_range_to_literal(nbegin, nend, "Complex")) {
      result = ndt::make_kind_sym(complex_kind);
    } else if (isupper(*nbegin)) {
      if (!parse_token_ds(begin, end, '[')) {
        result = ndt::typevar_type::make(std::string(nbegin, nend));
      } else {
        ndt::type arg_tp = parse_datashape(begin, end, symtable);
        if (arg_tp.is_null()) {
          throw datashape_parse_error(begin, "expected a dynd type");
        }
        if (!parse_token_ds(begin, end, ']')) {
          throw datashape_parse_error(begin, "expected closing ']'");
        }
        result = ndt::typevar_constructed_type::make(std::string(nbegin, nend), arg_tp);
      }
    } else {
      std::string n(nbegin, nend);
      const map<std::string, ndt::type> &bit = builtin_types();
      map<std::string, ndt::type>::const_iterator i = bit.find(n);
      if (i != bit.end()) {
        result = i->second;
      } else {
        i = symtable.find(n);
        if (i != symtable.end()) {
          result = i->second;
        } else {
          parse::skip_whitespace_and_pound_comments(rbegin, end);
          throw datashape_parse_error(rbegin, "unrecognized data type");
        }
      }
    }
  } else if (parse::parse_token_no_ws(begin, end, "...")) {
    // An unnamed ellipsis dim
    if (parse_token_ds(begin, end, '*')) { // ASTERISK
      ndt::type element_type = parse_datashape(begin, end, symtable);
      if (element_type.is_null()) {
        throw datashape_parse_error(begin, "expected a dynd type");
      }
      result = ndt::make_ellipsis_dim(element_type);
    } else {
      throw datashape_parse_error(begin, "expected a '*'");
    }
  }
  // struct
  if (result.is_null()) {
    result = parse_struct(begin, end, symtable);
  }
  // tuple or funcproto
  if (result.is_null()) {
    result = parse_tuple_or_funcproto(begin, end, symtable);
  }
  if (!result.is_null()) {
    rbegin = begin;
    return result;
  } else {
    return ndt::type();
  }
}

// This is what parses a single datashape as an ndt::type
//    datashape : datashape_nooption
//              | QUESTIONMARK datashape_nooption
static ndt::type parse_datashape(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  parse::skip_whitespace_and_pound_comments(begin, end);
  if (parse::parse_token_no_ws(begin, end, '?')) {
    ndt::type val_tp = parse_datashape_nooption(begin, end, symtable);
    if (!val_tp.is_null()) {
      rbegin = begin;
      return ndt::option_type::make(val_tp);
    } else {
      return ndt::type();
    }
  } else {
    return parse_datashape_nooption(rbegin, end, symtable);
  }
}

static ndt::type parse_stmt(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable)
{
  const char *begin = rbegin;
  // stmt : TYPE name EQUALS rhs_expression
  // NOTE that this doesn't support parameterized lhs_expression, this is subset
  // of Blaze datashape
  if (parse_token_ds(begin, end, "type")) {
    const map<std::string, ndt::type> &bit = builtin_types();
    const char *saved_begin = begin;
    const char *tname_begin, *tname_end;
    if (!parse::skip_required_whitespace(begin, end)) {
      if (begin == end) {
        // If it's only "type" by itself, return the "type" type
        rbegin = begin;
        return bit.find("type")->second;
      } else {
        return ndt::type();
      }
    }
    if (parse_token_ds(begin, end, '|')) {
      ndt::type pattern = parse_datashape(begin, end, symtable);
      if (pattern.is_null()) {
        throw datashape_parse_error(begin, "expected a data type");
      }
      rbegin = begin;
      return ndt::make_type(pattern);
    }
    if (!parse::parse_name_no_ws(begin, end, tname_begin, tname_end)) {
      parse::skip_whitespace_and_pound_comments(begin, end);
      if (begin == end) {
        // If it's only "type" by itself, return the "type" type
        rbegin = begin;
        return bit.find("type")->second;
      } else {
        throw datashape_parse_error(saved_begin, "expected an identifier for a type name");
      }
    }
    if (!parse_token_ds(begin, end, '=')) {
      throw datashape_parse_error(begin, "expected an '='");
    }
    ndt::type result = parse_datashape(begin, end, symtable);
    if (result.is_null()) {
      throw datashape_parse_error(begin, "expected a data type");
    }
    std::string tname(tname_begin, tname_end);
    // ACTION: Put the parsed type in the symbol table
    if (bit.find(tname) != bit.end()) {
      parse::skip_whitespace_and_pound_comments(saved_begin, end);
      throw datashape_parse_error(saved_begin, "cannot redefine a builtin type");
    }
    if (symtable.find(tname) != symtable.end()) {
      parse::skip_whitespace_and_pound_comments(saved_begin, end);
      throw datashape_parse_error(saved_begin, "type name already defined in datashape string");
    }
    symtable[tname] = result;
    rbegin = begin;
    return result;
  } else {
    // stmt : rhs_expression
    return parse_datashape(rbegin, end, symtable);
  }
}

// top : stmt stmt*
static ndt::type parse_top(const char *&begin, const char *end, map<std::string, ndt::type> &symtable)
{
  ndt::type result = parse_stmt(begin, end, symtable);
  if (result.is_null()) {
    throw datashape_parse_error(begin, "expected a datashape statement");
  }
  for (;;) {
    ndt::type next = parse_stmt(begin, end, symtable);
    if (next.is_null()) {
      parse::skip_whitespace_and_pound_comments(begin, end);
      if (begin != end) {
        throw datashape_parse_error(begin, "unexpected token in datashape");
      }
      return result;
    } else {
      result = next;
    }
  }
}

/**
 * Returns the row/column where the error occured, as well as the current and
 * previous
 * lines for printing some context.
 */
static void get_error_line_column(const char *begin, const char *end, const char *position, std::string &out_line_prev,
                                  std::string &out_line_cur, int &out_line, int &out_column)
{
  out_line_prev = "";
  out_line_cur = "";
  out_line = 1;
  while (begin < end) {
    const char *line_end = (const char *)memchr(begin, '\n', end - begin);
    out_line_prev.swap(out_line_cur);
    // If no \n was found
    if (line_end == NULL) {
      out_column = int(position - begin + 1);
      out_line_cur = std::string(begin, end);
      return;
    } else {
      out_line_cur = std::string(begin, line_end);
      ++line_end;
      if (position < line_end) {
        out_column = int(position - begin + 1);
        return;
      }
    }
    begin = line_end;
    ++out_line;
  }

  throw runtime_error("Cannot get line number of error, its position is out of range");
}

ndt::type dynd::type_from_datashape(const char *datashape_begin, const char *datashape_end)
{
  try
  {
    // Symbol table for intermediate types declared in the datashape
    map<std::string, ndt::type> symtable;
    // Parse the datashape and construct the type
    const char *begin = datashape_begin, *end = datashape_end;
    return parse_top(begin, end, symtable);
  }
  catch (const datashape_parse_error &e)
  {
    stringstream ss;
    std::string line_prev, line_cur;
    int line, column;
    get_error_line_column(datashape_begin, datashape_end, e.get_position(), line_prev, line_cur, line, column);
    ss << "Error parsing datashape at line " << line << ", column " << column << "\n";
    ss << "Message: " << e.get_message() << "\n";
    if (line > 1) {
      ss << line_prev << "\n";
    }
    ss << line_cur << "\n";
    for (int i = 0; i < column - 1; ++i) {
      ss << " ";
    }
    ss << "^\n";
    throw runtime_error(ss.str());
  }
}

nd::array dynd::parse_type_constr_args(const std::string &str)
{
  nd::array result;
  std::map<std::string, ndt::type> symtable;
  if (!str.empty()) {
    const char *begin = &str[0], *end = &str[0] + str.size();
    try
    {
      result = parse_type_constr_args(begin, end, symtable);
    }
    catch (const datashape_parse_error &e)
    {
      stringstream ss;
      std::string line_prev, line_cur;
      int line, column;
      get_error_line_column(&str[0], end, e.get_position(), line_prev, line_cur, line, column);
      ss << "Error parsing datashape at line " << line << ", column " << column << "\n";
      ss << "Message: " << e.get_message() << "\n";
      if (line > 1) {
        ss << line_prev << "\n";
      }
      ss << line_cur << "\n";
      for (int i = 0; i < column - 1; ++i) {
        ss << " ";
      }
      ss << "^\n";
      throw runtime_error(ss.str());
    }
  }
  if (result.is_null()) {
    stringstream ss;
    ss << "Cannot parse \"" << str << "\" as a dynd type";
    throw runtime_error(ss.str());
  }

  return result;
}
