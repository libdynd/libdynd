//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/json_formatter.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/option_type.hpp>

using namespace std;
using namespace dynd;

struct output_data {
  char *out_begin, *out_end, *out_capacity_end;
  memory_block_data::api *api;
  memory_block_data *blockref;
  bool struct_as_list;

  void ensure_capacity(intptr_t added_capacity)
  {
    // If there's not enough space, double the capacity
    if (out_capacity_end - out_end < added_capacity) {
      intptr_t current_size = out_end - out_begin;
      intptr_t new_capacity = 2 * (out_capacity_end - out_begin);
      // Make sure this adds the requested additional capacity
      if (new_capacity < current_size + added_capacity) {
        new_capacity = current_size + added_capacity;
      }
      out_begin = api->resize(blockref, out_begin, new_capacity);
      out_capacity_end = out_begin + new_capacity;
      out_end = out_begin + current_size;
    }
  }

  inline void write(char c)
  {
    ensure_capacity(1);
    *out_end++ = c;
  }

  // Write a literal string
  template <int N>
  inline void write(const char (&str)[N])
  {
    ensure_capacity(N - 1);
    memcpy(out_end, str, N - 1);
    out_end += N - 1;
  }

  // Write a std::string
  inline void write(const std::string &s)
  {
    ensure_capacity(s.size());
    memcpy(out_end, s.data(), s.size());
    out_end += s.size();
  }

  // Write a string-range
  inline void write(const char *begin, const char *end)
  {
    ensure_capacity(end - begin);
    memcpy(out_end, begin, end - begin);
    out_end += (end - begin);
  }
};

static void format_json(output_data &out, const ndt::type &dt, const char *arrmeta, const char *data);

static void format_json_bool(output_data &out, const ndt::type &dt, const char *arrmeta, const char *data)
{
  bool1 value(false);
  if (dt.get_type_id() == bool_type_id) {
    value = (*data != 0);
  } else {
    typed_data_assign(ndt::type::make<bool1>(), NULL, reinterpret_cast<char *>(&value), dt, arrmeta, data);
  }
  if (value) {
    out.write("true");
  } else {
    out.write("false");
  }
}

static void format_json_number(output_data &out, const ndt::type &dt, const char *arrmeta, const char *data)
{
  stringstream ss;
  dt.print_data(ss, arrmeta, data);
  out.write(ss.str());
}

static void print_escaped_unicode_codepoint(output_data &out, uint32_t cp, append_unicode_codepoint_t append_fn)
{
  if (cp < 0x80) {
    switch (cp) {
    case '\b':
      out.write("\\b");
      break;
    case '\f':
      out.write("\\f");
      break;
    case '\n':
      out.write("\\n");
      break;
    case '\r':
      out.write("\\r");
      break;
    case '\t':
      out.write("\\t");
      break;
    case '\\':
      out.write("\\\\");
      break;
    case '/':
      out.write("\\/");
      break;
    case '\"':
      out.write("\\\"");
      break;
    default:
      if (cp < 0x20 || cp == 0x7f) {
        stringstream ss;
        ss << "\\u";
        hexadecimal_print(ss, static_cast<uint16_t>(cp));
        out.write(ss.str());
      } else {
        out.write(static_cast<char>(cp));
      }
      break;
    }
  } else {
    out.ensure_capacity(16);
    append_fn(cp, out.out_end, out.out_capacity_end);
  }
  // TODO: Could have an ASCII output mode where unicode is always escaped
  /*
  else if (cp < 0x10000) {
      stringstream ss;
      ss << "\\u";
      hexadecimal_print(ss, static_cast<uint16_t>(cp));
      out.write(ss.str());
  } else {
      stringstream ss;
      ss << "\\U";
      hexadecimal_print(ss, static_cast<uint32_t>(cp));
      out.write(ss.str());
  }
  */
}

static void format_json_encoded_string(output_data &out, const char *begin, const char *end, string_encoding_t encoding)
{
  uint32_t cp;
  next_unicode_codepoint_t next_fn;
  append_unicode_codepoint_t append_fn;
  next_fn = get_next_unicode_codepoint_function(encoding, assign_error_nocheck);
  append_fn = get_append_unicode_codepoint_function(string_encoding_utf_8, assign_error_nocheck);
  out.write('\"');
  while (begin < end) {
    cp = next_fn(begin, end);
    print_escaped_unicode_codepoint(out, cp, append_fn);
  }
  out.write('\"');
}

static void format_json_string(output_data &out, const ndt::type &dt, const char *arrmeta, const char *data)
{
  const ndt::base_string_type *bsd = dt.extended<ndt::base_string_type>();
  string_encoding_t encoding = bsd->get_encoding();
  const char *begin = NULL, *end = NULL;
  bsd->get_string_range(&begin, &end, arrmeta, data);
  format_json_encoded_string(out, begin, end, encoding);
}

static void format_json_option(output_data &out, const ndt::type &dt, const char *arrmeta, const char *data)
{
  const ndt::option_type *ot = dt.extended<ndt::option_type>();
  if (ot->is_avail(arrmeta, data, &eval::default_eval_context)) {
    format_json(out, ot->get_value_type(), arrmeta, data);
  } else {
    out.write("null");
  }
}

static void format_json_datetime(output_data &out, const ndt::type &dt, const char *arrmeta, const char *data)
{
  switch (dt.get_type_id()) {
  case date_type_id:
  case datetime_type_id: {
    stringstream ss;
    dt.print_data(ss, arrmeta, data);
    std::string s = ss.str();
    format_json_encoded_string(out, s.data(), s.data() + s.size(), string_encoding_ascii);
    break;
  }
  default: {
    stringstream ss;
    ss << "Formatting dynd type " << dt << " as JSON is not implemented yet";
    throw runtime_error(ss.str());
  }
  }
}

static void format_json_type(output_data &out, const ndt::type &dt, const char *arrmeta, const char *data)
{
  switch (dt.get_type_id()) {
  case type_type_id: {
    stringstream ss;
    dt.print_data(ss, arrmeta, data);
    std::string s = ss.str();
    format_json_encoded_string(out, s.data(), s.data() + s.size(), string_encoding_ascii);
    break;
  }
  default: {
    stringstream ss;
    ss << "Formatting dynd type \"" << dt << "\" as JSON is not implemented yet";
    throw runtime_error(ss.str());
  }
  }
}

static void format_json_struct(output_data &out, const ndt::type &dt, const char *arrmeta, const char *data)
{
  const ndt::base_struct_type *bsd = dt.extended<ndt::base_struct_type>();
  intptr_t field_count = bsd->get_field_count();
  const size_t *data_offsets = bsd->get_data_offsets(arrmeta);
  const size_t *arrmeta_offsets = bsd->get_arrmeta_offsets_raw();

  if (out.struct_as_list) {
    out.write('[');
    for (intptr_t i = 0; i < field_count; ++i) {
      ::format_json(out, bsd->get_field_type(i), arrmeta + arrmeta_offsets[i], data + data_offsets[i]);
      if (i != field_count - 1) {
        out.write(',');
      }
    }
    out.write(']');
  } else {
    out.write('{');
    for (intptr_t i = 0; i < field_count; ++i) {
      const dynd::string &fname = bsd->get_field_name_raw(i);
      format_json_encoded_string(out, fname.begin(), fname.end(), string_encoding_utf_8);
      out.write(':');
      ::format_json(out, bsd->get_field_type(i), arrmeta + arrmeta_offsets[i], data + data_offsets[i]);
      if (i != field_count - 1) {
        out.write(',');
      }
    }
    out.write('}');
  }
}

static void format_json_dim(output_data &out, const ndt::type &dt, const char *arrmeta, const char *data)
{
  out.write('[');
  switch (dt.get_type_id()) {
  case fixed_dim_type_id: {
    const ndt::base_dim_type *sad = dt.extended<ndt::base_dim_type>();
    const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
    ndt::type element_tp = sad->get_element_type();
    intptr_t size = md->dim_size, stride = md->stride;
    arrmeta += sizeof(fixed_dim_type_arrmeta);
    for (intptr_t i = 0; i < size; ++i) {
      ::format_json(out, element_tp, arrmeta, data + i * stride);
      if (i != size - 1) {
        out.write(',');
      }
    }
    break;
  }
  case var_dim_type_id: {
    const ndt::var_dim_type *vad = dt.extended<ndt::var_dim_type>();
    const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
    const var_dim_type_data *d = reinterpret_cast<const var_dim_type_data *>(data);
    ndt::type element_tp = vad->get_element_type();
    intptr_t size = d->size, stride = md->stride;
    const char *begin = d->begin + md->offset;
    arrmeta += sizeof(var_dim_type_arrmeta);
    for (intptr_t i = 0; i < size; ++i) {
      ::format_json(out, element_tp, arrmeta, begin + i * stride);
      if (i != size - 1) {
        out.write(',');
      }
    }
    break;
  }
  default: {
    stringstream ss;
    ss << "Formatting dynd type " << dt << " as JSON is not implemented yet";
    throw runtime_error(ss.str());
  }
  }
  out.write(']');
}

static void format_json(output_data &out, const ndt::type &dt, const char *arrmeta, const char *data)
{
  switch (dt.get_kind()) {
  case bool_kind:
    format_json_bool(out, dt, arrmeta, data);
    break;
  case sint_kind:
  case uint_kind:
  case real_kind:
  case complex_kind:
    format_json_number(out, dt, arrmeta, data);
    break;
  case string_kind:
    format_json_string(out, dt, arrmeta, data);
    break;
  case datetime_kind:
    format_json_datetime(out, dt, arrmeta, data);
    break;
  case type_kind:
    format_json_type(out, dt, arrmeta, data);
    break;
  case struct_kind:
    format_json_struct(out, dt, arrmeta, data);
    break;
  case option_kind:
    format_json_option(out, dt, arrmeta, data);
    break;
  case dim_kind:
    format_json_dim(out, dt, arrmeta, data);
    break;
  default: {
    stringstream ss;
    ss << "Formatting dynd type " << dt << " as JSON is not implemented yet";
    throw runtime_error(ss.str());
  }
  }
}

nd::array dynd::format_json(const nd::array &n, bool struct_as_list)
{
  // Create a UTF-8 string
  nd::array result = nd::empty(ndt::string_type::make());

  // Initialize the output with some memory
  output_data out;
  out.blockref = reinterpret_cast<const string_type_arrmeta *>(result.get_arrmeta())->blockref;
  out.api = out.blockref->get_api();
  out.out_begin = out.api->allocate(out.blockref, 1024);
  out.out_capacity_end = out.out_begin + 1024;
  out.out_end = out.out_begin;
  out.struct_as_list = struct_as_list;

  if (!n.get_type().is_expression()) {
    ::format_json(out, n.get_type(), n.get_arrmeta(), n.get_readonly_originptr());
  } else {
    nd::array tmp = n.eval();
    ::format_json(out, tmp.get_type(), tmp.get_arrmeta(), tmp.get_readonly_originptr());
  }

  // Shrink the memory to fit, and set the pointers in the output
  string *d = reinterpret_cast<string *>(result.get_readwrite_originptr());
  char *begin = out.out_begin;
  char *end = out.out_capacity_end;
  begin = out.api->resize(out.blockref, begin, out.out_end - out.out_begin);
  end = begin + (out.out_end - out.out_begin);
  d->assign(begin, end - begin);

  // Finalize processing and mark the result as immutable
  result.get_type().extended()->arrmeta_finalize_buffers(result.get_arrmeta());
  result.flag_as_immutable();

  return result;
}
