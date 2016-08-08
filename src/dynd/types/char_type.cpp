//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/exceptions.hpp>
#include <dynd/parse_util.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/datashape_parser.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

uint32_t ndt::char_type::get_code_point(const char *data) const {
  next_unicode_codepoint_t next_fn;
  next_fn = get_next_unicode_codepoint_function(m_encoding, assign_error_nocheck);
  return next_fn(data, data + get_data_size());
}

void ndt::char_type::set_code_point(char *out_data, uint32_t cp) {
  append_unicode_codepoint_t append_fn;
  append_fn = get_append_unicode_codepoint_function(m_encoding, assign_error_nocheck);
  append_fn(cp, out_data, out_data + get_data_size());
}

void ndt::char_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const {
  // Print as an escaped string
  o << "\"";
  print_escaped_unicode_codepoint(o, get_code_point(data), false);
  o << "\"";
}

void ndt::char_type::print_type(std::ostream &o) const {

  o << "char";
  if (m_encoding != string_encoding_utf_32) {
    o << "['" << m_encoding << "']";
  }
}

ndt::type ndt::char_type::get_canonical_type() const {
  // The canonical char type is UTF-32
  if (m_encoding == string_encoding_utf_32) {
    return type(this, true);
  } else {
    return make_type<char_type>(string_encoding_utf_32);
  }
}

bool ndt::char_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp), const type &DYND_UNUSED(src_tp)) const {
  // Don't shortcut anything to 'nocheck' error checking, so that
  // decoding errors get caught appropriately.
  return false;
}

bool ndt::char_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else if (rhs.get_id() != char_id) {
    return false;
  } else {
    const char_type *dt = static_cast<const char_type *>(&rhs);
    return m_encoding == dt->m_encoding;
  }
}

// char_type : char | char[encoding]
ndt::type ndt::char_type::parse_type_args(type_id_t DYND_UNUSED(id), const char *&rbegin, const char *end,
                                          std::map<std::string, ndt::type> &DYND_UNUSED(symtable)) {
  const char *begin = rbegin;
  if (datashape::parse_token(begin, end, '[')) {
    const char *saved_begin = begin;
    std::string encoding_str;
    if (!datashape::parse_quoted_string(begin, end, encoding_str)) {
      throw datashape::internal_parse_error(saved_begin, "expected a string encoding");
    }
    string_encoding_t encoding;
    if (!encoding_str.empty()) {
      encoding = datashape::string_to_encoding(saved_begin, encoding_str);
    } else {
      throw datashape::internal_parse_error(begin, "expected string encoding");
    }
    if (!datashape::parse_token(begin, end, ']')) {
      throw datashape::internal_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::make_type<ndt::char_type>(encoding);
  } else {
    return ndt::make_type<ndt::char_type>();
  }
}