//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/string_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

ndt::string_type::string_type()
    : base_string_type(string_id, sizeof(string), alignof(string), type_flag_zeroinit | type_flag_destructor, 0)
{
}

void ndt::string_type::get_string_range(const char **out_begin, const char **out_end, const char *DYND_UNUSED(arrmeta),
                                        const char *data) const
{
  *out_begin = reinterpret_cast<const string *>(data)->begin();
  *out_end = reinterpret_cast<const string *>(data)->end();
}

void ndt::string_type::set_from_utf8_string(const char *DYND_UNUSED(arrmeta), char *dst, const char *utf8_begin,
                                            const char *utf8_end, const eval::eval_context *ectx) const
{
  assign_error_mode errmode = ectx->errmode;
  const intptr_t src_charsize = 1;
  intptr_t dst_charsize = string_encoding_char_size_table[string_encoding_utf_8];
  char *dst_current;
  next_unicode_codepoint_t next_fn = get_next_unicode_codepoint_function(string_encoding_utf_8, errmode);
  append_unicode_codepoint_t append_fn = get_append_unicode_codepoint_function(string_encoding_utf_8, errmode);
  uint32_t cp;

  // Allocate the initial output as the src number of characters + some padding
  // TODO: Don't add padding if the output is not a multi-character encoding
  string dst_d;
  dst_d.resize(((utf8_end - utf8_begin) / src_charsize + 16) * dst_charsize * 1124 / 1024);
  char *dst_begin = dst_d.begin();
  char *dst_end = dst_d.end();

  dst_current = dst_begin;
  while (utf8_begin < utf8_end) {
    cp = next_fn(utf8_begin, utf8_end);
    // Append the codepoint, or increase the allocated memory as necessary
    if (dst_end - dst_current >= 8) {
      append_fn(cp, dst_current, dst_end);
    }
    else {
      char *dst_begin_saved = dst_begin;
      dst_d.resize(2 * dst_d.size());
      dst_begin = dst_d.begin();
      dst_end = dst_d.end();
      dst_current = dst_begin + (dst_current - dst_begin_saved);

      append_fn(cp, dst_current, dst_end);
    }
  }

  // Set the output
  reinterpret_cast<string *>(dst)->assign(dst_d.begin(), dst_current - dst_begin);
}

void ndt::string_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  uint32_t cp;
  next_unicode_codepoint_t next_fn;
  next_fn = get_next_unicode_codepoint_function(string_encoding_utf_8, assign_error_nocheck);
  const char *begin = reinterpret_cast<const string *>(data)->begin();
  const char *end = reinterpret_cast<const string *>(data)->end();

  // Print as an escaped string
  o << "\"";
  while (begin < end) {
    cp = next_fn(begin, end);
    print_escaped_unicode_codepoint(o, cp, false);
  }
  o << "\"";
}

void ndt::string_type::print_type(std::ostream &o) const { o << "string"; }

bool ndt::string_type::is_unique_data_owner(const char *DYND_UNUSED(arrmeta)) const { return true; }

ndt::type ndt::string_type::get_canonical_type() const { return type(this, true); }

void ndt::string_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *DYND_UNUSED(arrmeta),
                                 const char *DYND_UNUSED(data)) const
{
  out_shape[i] = -1;
  if (i + 1 < ndim) {
    stringstream ss;
    ss << "requested too many dimensions from type " << type(this, true);
    throw runtime_error(ss.str());
  }
}

bool ndt::string_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp), const type &DYND_UNUSED(src_tp)) const
{
  // Don't shortcut anything to 'nocheck' error checking, so that
  // decoding errors get caught appropriately.
  return false;
}

bool ndt::string_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_id() != string_id) {
    return false;
  }
  else {
    return true;
  }
}

void ndt::string_type::arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
                                           const std::string &DYND_UNUSED(indent)) const {}

void ndt::string_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const
{
  reinterpret_cast<string *>(data)->~string();
}

void ndt::string_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *data, intptr_t stride,
                                             size_t count) const
{
  for (size_t i = 0; i != count; ++i) {
    reinterpret_cast<string *>(data)->~string();
    data += stride;
  }
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::string_type::get_dynamic_type_properties() const
{
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["encoding"] = {ndt::type("string"), reinterpret_cast<const char *>(&m_encoding_repr)};

  return properties;
}
