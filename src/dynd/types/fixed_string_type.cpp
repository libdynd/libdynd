//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/exceptions.hpp>
#include <dynd/parse_util.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/types/datashape_parser.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

void ndt::fixed_string_type::get_string_range(const char **out_begin, const char **out_end,
                                              const char *DYND_UNUSED(arrmeta), const char *data) const
{
  // Beginning of the string
  *out_begin = data;

  switch (string_encoding_char_size_table[m_encoding]) {
  case 1: {
    const char *end = reinterpret_cast<const char *>(memchr(data, 0, get_data_size()));
    if (end != NULL) {
      *out_end = end;
    }
    else {
      *out_end = data + get_data_size();
    }
    break;
  }
  case 2: {
    const uint16_t *ptr = reinterpret_cast<const uint16_t *>(data);
    const uint16_t *ptr_max = ptr + get_data_size() / sizeof(uint16_t);
    while (ptr < ptr_max && *ptr != 0) {
      ++ptr;
    }
    *out_end = reinterpret_cast<const char *>(ptr);
    break;
  }
  case 4: {
    const uint32_t *ptr = reinterpret_cast<const uint32_t *>(data);
    const uint32_t *ptr_max = ptr + get_data_size() / sizeof(uint32_t);
    while (ptr < ptr_max && *ptr != 0) {
      ++ptr;
    }
    *out_end = reinterpret_cast<const char *>(ptr);
    break;
  }
  }
}

void ndt::fixed_string_type::set_from_utf8_string(const char *DYND_UNUSED(arrmeta), char *dst, const char *utf8_begin,
                                                  const char *utf8_end, const eval::eval_context *ectx) const
{
  assign_error_mode errmode = ectx->errmode;
  char *dst_end = dst + get_data_size();
  next_unicode_codepoint_t next_fn = get_next_unicode_codepoint_function(string_encoding_utf_8, errmode);
  append_unicode_codepoint_t append_fn = get_append_unicode_codepoint_function(m_encoding, errmode);
  uint32_t cp;

  while (utf8_begin < utf8_end && dst < dst_end) {
    cp = next_fn(utf8_begin, utf8_end);
    append_fn(cp, dst, dst_end);
  }
  if (utf8_begin < utf8_end) {
    if (errmode != assign_error_nocheck) {
      throw std::runtime_error("Input is too large to convert to "
                               "destination fixed-size string");
    }
  }
  else if (dst < dst_end) {
    memset(dst, 0, dst_end - dst);
  }
}

void ndt::fixed_string_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  uint32_t cp;
  next_unicode_codepoint_t next_fn;
  next_fn = get_next_unicode_codepoint_function(m_encoding, assign_error_nocheck);
  const char *data_end = data + get_data_size();

  // Print as an escaped string
  o << "\"";
  while (data < data_end) {
    cp = next_fn(data, data_end);
    if (cp != 0) {
      print_escaped_unicode_codepoint(o, cp, false);
    }
    else {
      break;
    }
  }
  o << "\"";
}

void ndt::fixed_string_type::print_type(std::ostream &o) const
{
  o << "fixed_string[" << m_stringsize;
  if (m_encoding != string_encoding_utf_8) {
    o << ", '" << m_encoding << "'";
  }
  o << "]";
}

ndt::type ndt::fixed_string_type::get_canonical_type() const { return type(this, true); }

bool ndt::fixed_string_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp),
                                                    const type &DYND_UNUSED(src_tp)) const
{
  // Don't shortcut anything to 'nocheck' error checking, so that
  // decoding errors get caught appropriately.
  return false;
}

bool ndt::fixed_string_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != fixed_string_id) {
    return false;
  }
  else {
    const fixed_string_type *dt = static_cast<const fixed_string_type *>(&rhs);
    return m_encoding == dt->m_encoding && m_stringsize == dt->m_stringsize;
  }
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::fixed_string_type::get_dynamic_type_properties() const
{
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["encoding"] = {ndt::type("string"), reinterpret_cast<const char *>(&m_encoding_repr)};

  return properties;
}

// fixed_string_type : fixed_string[NUMBER] |
//                     fixed_string[NUMBER,'encoding']
ndt::type ndt::fixed_string_type::parse_type_args(type_id_t id, const char *&rbegin, const char *end,
                                                  std::map<std::string, ndt::type> &DYND_UNUSED(symtable)) {
  const char *begin = rbegin;
  if (datashape::parse_token(begin, end, '[')) {
    const char *saved_begin = begin;
    std::string value = datashape::parse_number(begin, end);
    std::string encoding_str;
    string_encoding_t encoding = string_encoding_utf_8;
    int string_size = 0;
    if (!value.empty()) {
      string_size = atoi(value.c_str());
      if (string_size == 0) {
        throw datashape::internal_parse_error(saved_begin, "string size cannot be zero");
      }
      if (datashape::parse_token(begin, end, ',')) {
        saved_begin = begin;
        if (!datashape::parse_quoted_string(begin, end, encoding_str)) {
          throw datashape::internal_parse_error(saved_begin, "expected a string encoding");
        }
        encoding = datashape::string_to_encoding(saved_begin, encoding_str);
      }
    } else {
      if (!datashape::parse_quoted_string(begin, end, encoding_str)) {
        throw datashape::internal_parse_error(saved_begin, "expected a size integer");
      }
      encoding = datashape::string_to_encoding(saved_begin, encoding_str);
    }
    if (!datashape::parse_token(begin, end, ']')) {
      throw datashape::internal_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::make_type<ndt::fixed_string_type>(string_size, encoding);
  }

  throw datashape::internal_parse_error(begin, "expected opening '['");
}