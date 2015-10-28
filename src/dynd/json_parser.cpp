//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/json_parser.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/kernels/string_numeric_assignment_kernels.hpp>
#include <dynd/parser_util.hpp>

using namespace std;
using namespace dynd;

namespace {
class json_parse_error : public parse::parse_error {
  ndt::type m_type;

public:
  json_parse_error(const char *position, const std::string &message, const ndt::type &tp)
      : parse::parse_error(position, message), m_type(tp)
  {
  }
  virtual ~json_parse_error() throw()
  {
  }
  const ndt::type &get_type() const
  {
    return m_type;
  }
};
} // anonymous namespace

nd::array nd::json::parse(const ndt::type &tp, const std::string &str)
{
  return parse_json(tp, str.data());
}

static void json_as_buffer(const nd::array &json, nd::array &out_tmp_ref, const char *&begin, const char *&end)
{
  // Check the type of 'json', and get pointers to the begin/end of a UTF-8
  // buffer
  ndt::type json_type = json.get_type().value_type();
  switch (json_type.get_kind()) {
  case string_kind: {
    const ndt::base_string_type *sdt = json_type.extended<ndt::base_string_type>();
    switch (sdt->get_encoding()) {
    case string_encoding_ascii:
    case string_encoding_utf_8:
      out_tmp_ref = json.eval();
      // The data is already UTF-8, so use the buffer directly
      sdt->get_string_range(&begin, &end, out_tmp_ref.get_arrmeta(), out_tmp_ref.get_readonly_originptr());
      break;
    default: {
      // The data needs to be converted to UTF-8 before parsing
      ndt::type utf8_tp = ndt::string_type::make();
      out_tmp_ref = json.ucast(utf8_tp).eval();
      sdt = static_cast<const ndt::base_string_type *>(utf8_tp.extended());
      sdt->get_string_range(&begin, &end, out_tmp_ref.get_arrmeta(), out_tmp_ref.get_readonly_originptr());
      break;
    }
    }
    break;
  }
  case bytes_kind: {
    out_tmp_ref = json.eval();
    const ndt::base_bytes_type *bdt = json_type.extended<ndt::base_bytes_type>();
    bdt->get_bytes_range(&begin, &end, out_tmp_ref.get_arrmeta(), out_tmp_ref.get_readonly_originptr());
    break;
  }
  default: {
    stringstream ss;
    ss << "Input for JSON parsing must be either bytes (interpreted as UTF-8) "
          "or a string, not \"" << json_type << "\"";
    throw runtime_error(ss.str());
    break;
  }
  }
}

void dynd::parse_json(nd::array &out, const nd::array &json, const eval::eval_context *ectx)
{
  const char *json_begin = NULL, *json_end = NULL;
  nd::array tmp_ref;
  json_as_buffer(json, tmp_ref, json_begin, json_end);
  parse_json(out, json_begin, json_end, ectx);
}

nd::array dynd::parse_json(const ndt::type &tp, const nd::array &json, const eval::eval_context *ectx)
{
  const char *json_begin = NULL, *json_end = NULL;
  nd::array tmp_ref;
  json_as_buffer(json, tmp_ref, json_begin, json_end);
  return parse_json(tp, json_begin, json_end, ectx);
}

static void parse_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&json_begin,
                       const char *json_end, const eval::eval_context *ectx);

static const char *skip_whitespace(const char *begin, const char *end)
{
  while (begin < end && isspace(*begin)) {
    ++begin;
  }

  return begin;
}

template <int N>
static bool parse_token(const char *&begin, const char *end, const char (&token)[N])
{
  const char *begin_skipws = skip_whitespace(begin, end);
  if (N - 1 <= end - begin_skipws && memcmp(begin_skipws, token, N - 1) == 0) {
    begin = begin_skipws + N - 1;
    return true;
  } else {
    return false;
  }
}

static void skip_json_value(const char *&begin, const char *end)
{
  begin = skip_whitespace(begin, end);
  if (begin == end) {
    throw parse::parse_error(begin, "malformed JSON, expecting an element");
  }
  char c = *begin;
  switch (c) {
  // Object
  case '{':
    ++begin;
    if (!parse_token(begin, end, "}")) {
      for (;;) {
        const char *strbegin, *strend;
        bool escaped;
        begin = skip_whitespace(begin, end);
        if (!parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
          throw parse::parse_error(begin, "expected string for name in object dict");
        }
        if (!parse_token(begin, end, ":")) {
          throw parse::parse_error(begin, "expected ':' separating name from value in object dict");
        }
        skip_json_value(begin, end);
        if (!parse_token(begin, end, ",")) {
          break;
        }
      }
      if (!parse_token(begin, end, "}")) {
        throw parse::parse_error(begin, "expected object separator ',' or terminator '}'");
      }
    }
    break;
  // Array
  case '[':
    ++begin;
    if (!parse_token(begin, end, "]")) {
      for (;;) {
        skip_json_value(begin, end);
        if (!parse_token(begin, end, ",")) {
          break;
        }
      }
      if (!parse_token(begin, end, "]")) {
        throw parse::parse_error(begin, "expected array separator ',' or terminator ']'");
      }
    }
    break;
  case '"': {
    const char *strbegin, *strend;
    bool escaped;
    if (!parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
      throw parse::parse_error(begin, "invalid string");
    }
    break;
  }
  case 't':
    if (!parse_token(begin, end, "true")) {
      throw parse::parse_error(begin, "invalid json value");
    }
    break;
  case 'f':
    if (!parse_token(begin, end, "false")) {
      throw parse::parse_error(begin, "invalid json value");
    }
    break;
  case 'n':
    if (!parse_token(begin, end, "null")) {
      throw parse::parse_error(begin, "invalid json value");
    }
    break;
  default:
    if (c == '-' || ('0' <= c && c <= '9')) {
      const char *nbegin = NULL, *nend = NULL;
      if (!parse::parse_json_number_no_ws(begin, end, nbegin, nend)) {
        throw parse::parse_error(begin, "invalid number");
      }
    } else {
      throw parse::parse_error(begin, "invalid json value");
    }
  }
}

static void parse_strided_dim_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&begin,
                                   const char *end, const eval::eval_context *ectx)
{
  intptr_t dim_size, stride;
  ndt::type el_tp;
  const char *el_arrmeta;
  if (!tp.get_as_strided(arrmeta, &dim_size, &stride, &el_tp, &el_arrmeta)) {
    throw json_parse_error(begin, "expected a strided dimension", tp);
  }

  if (!parse_token(begin, end, "[")) {
    throw json_parse_error(begin, "expected list starting with '['", tp);
  }
  for (intptr_t i = 0; i < dim_size; ++i) {
    parse_json(el_tp, el_arrmeta, out_data + i * stride, begin, end, ectx);
    if (i < dim_size - 1 && !parse_token(begin, end, ",")) {
      throw json_parse_error(begin, "array is too short, expected ',' list item separator", tp);
    }
  }
  if (!parse_token(begin, end, "]")) {
    throw json_parse_error(begin, "array is too long, expected list terminator ']'", tp);
  }
}

static void parse_var_dim_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&begin,
                               const char *end, const eval::eval_context *ectx)
{
  const ndt::var_dim_type *vad = tp.extended<ndt::var_dim_type>();
  const var_dim_type_arrmeta *md = reinterpret_cast<const var_dim_type_arrmeta *>(arrmeta);
  intptr_t stride = md->stride;
  const ndt::type &element_tp = vad->get_element_type();

  var_dim_type_data *out = reinterpret_cast<var_dim_type_data *>(out_data);

  memory_block_data::api *allocator = md->blockref->get_api();
  intptr_t size = 0, allocated_size = 8;
  out->begin = allocator->allocate(md->blockref, allocated_size);

  if (!parse_token(begin, end, "[")) {
    throw json_parse_error(begin, "expected array starting with '['", tp);
  }
  // If it's not an empty list, start the loop parsing the elements
  if (!parse_token(begin, end, "]")) {
    for (;;) {
      // Increase the allocated array size if necessary
      if (size == allocated_size) {
        allocated_size *= 2;
        out->begin = allocator->resize(md->blockref, out->begin, allocated_size);
      }
      ++size;
      out->size = size;
      parse_json(element_tp, arrmeta + sizeof(var_dim_type_arrmeta), out->begin + (size - 1) * stride, begin, end,
                 ectx);
      if (!parse_token(begin, end, ",")) {
        break;
      }
    }
    if (!parse_token(begin, end, "]")) {
      throw json_parse_error(begin, "expected array separator ',' or terminator ']'", tp);
    }
  }

  // Shrink-wrap the memory to just fit the string
  out->begin = allocator->resize(md->blockref, out->begin, size);
  out->size = size;
}

static bool parse_struct_json_from_object(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&begin,
                                          const char *end, const eval::eval_context *ectx)
{
  const char *saved_begin = begin;
  if (!parse_token(begin, end, "{")) {
    return false;
  }

  const ndt::base_struct_type *fsd = tp.extended<ndt::base_struct_type>();
  intptr_t field_count = fsd->get_field_count();
  const size_t *data_offsets = fsd->get_data_offsets(arrmeta);
  const size_t *arrmeta_offsets = fsd->get_arrmeta_offsets_raw();

  // Keep track of which fields we've seen
  shortvector<bool> populated_fields(field_count);
  memset(populated_fields.get(), 0, sizeof(bool) * field_count);

  // If it's not an empty object, start the loop parsing the elements
  if (!parse_token(begin, end, "}")) {
    for (;;) {
      const char *strbegin, *strend;
      bool escaped;
      begin = skip_whitespace(begin, end);
      if (!parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
        throw json_parse_error(begin, "expected string for name in object dict", tp);
      }
      if (!parse_token(begin, end, ":")) {
        throw json_parse_error(begin, "expected ':' separating name from value in object dict", tp);
      }
      intptr_t i;
      if (escaped) {
        std::string name;
        parse::unescape_string(strbegin, strend, name);
        i = fsd->get_field_index(name);
      } else {
        i = fsd->get_field_index(strbegin, strend);
      }
      if (i == -1) {
        // TODO: Add an error policy to this parser of whether to throw an error
        //       or not. For now, just throw away fields not in the destination.
        skip_json_value(begin, end);
      } else {
        parse_json(fsd->get_field_type(i), arrmeta + arrmeta_offsets[i], out_data + data_offsets[i], begin, end, ectx);
        populated_fields[i] = true;
      }
      if (!parse_token(begin, end, ",")) {
        break;
      }
    }
    if (!parse_token(begin, end, "}")) {
      throw json_parse_error(begin, "expected object dict separator ',' or terminator '}'", tp);
    }
  }

  for (intptr_t i = 0; i < field_count; ++i) {
    if (!populated_fields[i]) {
      const ndt::type &field_tp = fsd->get_field_type(i);
      if (field_tp.get_type_id() == option_type_id) {
        field_tp.extended<ndt::option_type>()->assign_na(arrmeta + arrmeta_offsets[i], out_data + data_offsets[i],
                                                         &eval::default_eval_context);
      } else {
        stringstream ss;
        ss << "object dict does not contain the field ";
        print_escaped_utf8_string(ss, fsd->get_field_name(i));
        ss << " as required by the data type";
        throw json_parse_error(skip_whitespace(saved_begin, end), ss.str(), tp);
      }
    }
  }

  return true;
}

static bool parse_tuple_json_from_list(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&begin,
                                       const char *end, const eval::eval_context *ectx)
{
  if (!parse_token(begin, end, "[")) {
    return false;
  }

  auto fsd = tp.extended<ndt::base_tuple_type>();
  intptr_t field_count = fsd->get_field_count();
  const size_t *data_offsets = fsd->get_data_offsets(arrmeta);
  const size_t *arrmeta_offsets = fsd->get_arrmeta_offsets_raw();

  // Loop through all the fields
  for (intptr_t i = 0; i != field_count; ++i) {
    begin = skip_whitespace(begin, end);
    parse_json(fsd->get_field_type(i), arrmeta + arrmeta_offsets[i], out_data + data_offsets[i], begin, end, ectx);
    if (i != field_count - 1 && !parse_token(begin, end, ",")) {
      throw json_parse_error(begin, "expected list item separator ','", tp);
    }
  }

  if (!parse_token(begin, end, "]")) {
    throw json_parse_error(begin, "expected end of list ']'", tp);
  }

  return true;
}

static void parse_struct_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&begin,
                              const char *end, const eval::eval_context *ectx)
{
  if (parse_struct_json_from_object(tp, arrmeta, out_data, begin, end, ectx)) {
  } else if (parse_tuple_json_from_list(tp, arrmeta, out_data, begin, end, ectx)) {
  } else {
    throw json_parse_error(begin, "expected object dict starting with '{' or list with '['", tp);
  }
}

static void parse_tuple_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&begin,
                             const char *end, const eval::eval_context *ectx)
{
  if (parse_tuple_json_from_list(tp, arrmeta, out_data, begin, end, ectx)) {
  } else {
    throw json_parse_error(begin, "expected object dict starting with '{' or list with '['", tp);
  }
}

static void parse_bool_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&rbegin,
                            const char *end, bool option, const eval::eval_context *ectx)
{
  const char *begin = rbegin;
  char value = 3;
  const char *nbegin, *nend;
  bool escaped;
  if (parse_token(begin, end, "true")) {
    value = 1;
  } else if (parse_token(begin, end, "false")) {
    value = 0;
  } else if (parse_token(begin, end, "null")) {
    if (option || ectx->errmode != assign_error_nocheck) {
      value = 2;
    } else {
      value = 0;
    }
  } else if (parse::parse_json_number_no_ws(begin, end, nbegin, nend)) {
    if (nend - nbegin == 1) {
      if (*nbegin == '0') {
        value = 0;
      } else if (*nbegin == '1' || ectx->errmode == assign_error_nocheck) {
        value = 1;
      }
    }
  } else if (parse::parse_doublequote_string_no_ws(begin, end, nbegin, nend, escaped)) {
    if (!escaped) {
      parse::string_to_bool(&value, nbegin, nend, option, ectx->errmode);
    } else {
      std::string s;
      parse::unescape_string(nbegin, nend, s);
      parse::string_to_bool(&value, s.data(), s.data() + s.size(), option, ectx->errmode);
    }
  }

  if (value < 2) {
    if (tp.get_type_id() == bool_type_id) {
      *out_data = value;
    } else {
      typed_data_assign(tp, arrmeta, out_data, ndt::type::make<bool1>(), NULL, &value);
    }
    rbegin = begin;
  } else if (value == 2 && option) {
    if (!tp.is_expression()) {
      *out_data = value;
    } else {
      typed_data_assign(tp, arrmeta, out_data, ndt::option_type::make(ndt::type::make<bool1>()), NULL, &value);
    }
    rbegin = begin;
  } else {
    throw json_parse_error(rbegin, "expected a boolean true or false", tp);
  }
}

static void parse_number_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&rbegin,
                              const char *end, bool option, const eval::eval_context *ectx)
{
  const char *begin = rbegin;
  const char *nbegin, *nend;
  bool escaped = false;
  if (option && parse::parse_token_no_ws(begin, end, "null")) {
    ndt::option_type::make(tp).extended<ndt::option_type>()->assign_na(arrmeta, out_data, ectx);
  } else if (parse::parse_json_number_no_ws(begin, end, nbegin, nend)) {
    parse::string_to_number(out_data, tp.get_type_id(), nbegin, nend, false, ectx->errmode);
  } else if (parse::parse_doublequote_string_no_ws(begin, end, nbegin, nend, escaped)) {
    // Interpret the data inside the string as an int
    try
    {
      if (!escaped) {
        parse::string_to_number(out_data, tp.get_type_id(), nbegin, nend, option, ectx->errmode);
      } else {
        std::string s;
        parse::unescape_string(nbegin, nend, s);
        parse::string_to_number(out_data, tp.get_type_id(), nbegin, nend, option, ectx->errmode);
      }
    }
    catch (const std::exception &e)
    {
      throw json_parse_error(rbegin, e.what(), tp);
    }
    catch (const dynd::dynd_exception &e)
    {
      throw json_parse_error(rbegin, e.what(), tp);
    }
  } else {
    throw json_parse_error(rbegin, "expected a number", tp);
  }
  rbegin = begin;
}

static void parse_string_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&rbegin,
                              const char *end, const eval::eval_context *ectx)
{
  const char *begin = rbegin;
  begin = skip_whitespace(begin, end);
  const char *strbegin, *strend;
  bool escaped;
  if (parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
    const ndt::base_string_type *bsd = tp.extended<ndt::base_string_type>();
    try
    {
      if (!escaped) {
        bsd->set_from_utf8_string(arrmeta, out_data, strbegin, strend, ectx);
      } else {
        std::string val;
        parse::unescape_string(strbegin, strend, val);
        bsd->set_from_utf8_string(arrmeta, out_data, val, ectx);
      }
    }
    catch (const std::exception &e)
    {
      throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
    }
    catch (const dynd::dynd_exception &e)
    {
      throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
    }
  } else {
    throw json_parse_error(begin, "expected a string", tp);
  }
  rbegin = begin;
}

static void parse_datetime_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&rbegin,
                                const char *end, bool option, const eval::eval_context *ectx)
{
  const char *begin = rbegin;
  begin = skip_whitespace(begin, end);
  const char *strbegin, *strend;
  bool escaped;
  if (option && parse_token(begin, end, "null")) {
    switch (tp.get_type_id()) {
    case date_type_id:
      *reinterpret_cast<int32_t *>(out_data) = DYND_DATE_NA;
      return;
    case time_type_id:
      *reinterpret_cast<int64_t *>(out_data) = DYND_TIME_NA;
      return;
    case datetime_type_id:
      *reinterpret_cast<int64_t *>(out_data) = DYND_DATETIME_NA;
      return;
    default:
      break;
    }
    stringstream ss;
    ss << "Unrecognized datetime type " << tp;
    throw runtime_error(ss.str());
  } else if (parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
    std::string val;
    if (escaped) {
      parse::unescape_string(strbegin, strend, val);
      strbegin = val.data();
      strend = strbegin + val.size();
    }
    try
    {
      tp.extended()->set_from_utf8_string(arrmeta, out_data, strbegin, strend, ectx);
    }
    catch (const std::exception &e)
    {
      throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
    }
    catch (const dynd::dynd_exception &e)
    {
      throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
    }
  } else {
    throw json_parse_error(begin, "expected a string", tp);
  }
  rbegin = begin;
}

static void parse_type(const ndt::type &tp, const char *DYND_UNUSED(arrmeta), char *out_data, const char *&rbegin,
                       const char *end, bool option, const eval::eval_context *DYND_UNUSED(ectx))
{
  const char *begin = rbegin;
  begin = skip_whitespace(begin, end);
  const char *strbegin, *strend;
  bool escaped;
  if (option && parse_token(begin, end, "null")) {
    switch (tp.get_type_id()) {
    case type_type_id:
      *reinterpret_cast<ndt::type *>(out_data) = ndt::type();
      return;
    default:
      break;
    }
    stringstream ss;
    ss << "Unrecognized type type \"" << tp << "\"";
    throw runtime_error(ss.str());
  } else if (parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
    std::string val;
    if (escaped) {
      parse::unescape_string(strbegin, strend, val);
      strbegin = val.data();
      strend = strbegin + val.size();
    }
    try
    {
      *reinterpret_cast<ndt::type *>(out_data) = ndt::type(strbegin, strend);
    }
    catch (const std::exception &e)
    {
      throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
    }
    catch (const dynd::dynd_exception &e)
    {
      throw json_parse_error(skip_whitespace(rbegin, begin), e.what(), tp);
    }
  } else {
    throw json_parse_error(begin, "expected a string", tp);
  }
  rbegin = begin;
}

static void parse_dim_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&begin,
                           const char *end, const eval::eval_context *ectx)
{
  switch (tp.get_type_id()) {
  case fixed_dim_type_id:
    parse_strided_dim_json(tp, arrmeta, out_data, begin, end, ectx);
    break;
  case var_dim_type_id:
    parse_var_dim_json(tp, arrmeta, out_data, begin, end, ectx);
    break;
  default: {
    stringstream ss;
    ss << "parse_json: unsupported dynd dimension type \"" << tp << "\"";
    throw runtime_error(ss.str());
  }
  }
}

static void parse_option_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&begin,
                              const char *end, const eval::eval_context *ectx)
{
  begin = skip_whitespace(begin, end);
  const char *saved_begin = begin;
  if (tp.is_scalar()) {
    if (parse_token(begin, end, "null")) {
      tp.extended<ndt::option_type>()->assign_na(arrmeta, out_data, ectx);
      return;
    } else {
      const ndt::type &value_tp = tp.extended<ndt::option_type>()->get_value_type();
      const char *strbegin, *strend;
      bool escaped;
      if (parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
        try
        {
          if (!escaped) {
            tp.extended()->set_from_utf8_string(arrmeta, out_data, strbegin, strend, ectx);
          } else {
            std::string val;
            parse::unescape_string(strbegin, strend, val);
            tp.extended()->set_from_utf8_string(arrmeta, out_data, val, ectx);
          }
          return;
        }
        catch (const exception &e)
        {
          throw json_parse_error(saved_begin, e.what(), tp);
        }
        catch (const dynd::dynd_exception &e)
        {
          throw json_parse_error(saved_begin, e.what(), tp);
        }
      } else if (value_tp.get_kind() == bool_kind) {
        if (parse_token(begin, end, "true")) {
          *out_data = 1;
        } else if (parse_token(begin, end, "false")) {
          *out_data = 0;
        } else if (parse::parse_json_number_no_ws(begin, end, strbegin, strend)) {
          if (parse::compare_range_to_literal(strbegin, strend, "1")) {
            *out_data = 1;
          } else if (parse::compare_range_to_literal(strbegin, strend, "0")) {
            *out_data = 0;
          } else {
            throw json_parse_error(begin, "expected a boolean", tp);
          }
        } else {
          throw json_parse_error(begin, "expected a boolean", tp);
        }
        return;
      } else if (value_tp.get_kind() == sint_kind || value_tp.get_kind() == uint_kind ||
                 value_tp.get_kind() == real_kind || value_tp.get_kind() == complex_kind) {
        if (parse::parse_json_number_no_ws(begin, end, strbegin, strend)) {
          parse::string_to_number(out_data, value_tp.get_type_id(), strbegin, strend, false, ectx->errmode);
        } else {
          throw json_parse_error(begin, "expected a number", tp);
        }
        return;
      } else {
        throw json_parse_error(begin, "expected a string", tp);
      }
    }
  }

  stringstream ss;
  ss << "parse_json: unsupported dynd type \"" << tp << "\"";
  throw runtime_error(ss.str());
}

static void parse_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&begin, const char *end,
                       const eval::eval_context *ectx)
{
  begin = skip_whitespace(begin, end);
  switch (tp.get_kind()) {
  case dim_kind:
    parse_dim_json(tp, arrmeta, out_data, begin, end, ectx);
    return;
  case struct_kind:
    parse_struct_json(tp, arrmeta, out_data, begin, end, ectx);
    return;
  case tuple_kind:
    parse_tuple_json(tp, arrmeta, out_data, begin, end, ectx);
    return;
  case bool_kind:
    parse_bool_json(tp, arrmeta, out_data, begin, end, false, ectx);
    return;
  case sint_kind:
  case uint_kind:
  case real_kind:
  case complex_kind:
    parse_number_json(tp, arrmeta, out_data, begin, end, false, ectx);
    return;
  case string_kind:
    parse_string_json(tp, arrmeta, out_data, begin, end, ectx);
    return;
  case datetime_kind:
    parse_datetime_json(tp, arrmeta, out_data, begin, end, false, ectx);
    return;
  case type_kind:
    parse_type(tp, arrmeta, out_data, begin, end, false, ectx);
    return;
  case option_kind:
    parse_option_json(tp, arrmeta, out_data, begin, end, ectx);
    return;
  default:
    break;
  }

  stringstream ss;
  ss << "parse_json: unsupported dynd type \"" << tp << "\"";
  throw runtime_error(ss.str());
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

void print_json_parse_error_marker(std::ostream &o, const std::string &line_prev, const std::string &line_cur, int line,
                                   int column)
{
  if (line_cur.size() < 200) {
    // If the line is short, print the lines and indicate the error
    if (line > 1) {
      o << line_prev << "\n";
    }
    o << line_cur << "\n";
    for (int i = 0; i < column - 1; ++i) {
      o << " ";
    }
    o << "^\n";
  } else {
    // If the line is long, print a part of the line and indicate the error
    if (column < 80) {
      o << line_cur.substr(0, 80) << " ...\n";
      for (int i = 0; i < column - 1; ++i) {
        o << " ";
      }
      o << "^\n";
    } else {
      int start = column - 60;
      o << " ... " << line_cur.substr(start - 1, 80) << " ...\n";
      for (int i = 0; i < 65; ++i) {
        o << " ";
      }
      o << "^\n";
    }
  }
}

void dynd::validate_json(const char *json_begin, const char *json_end)
{
  try
  {
    const char *begin = json_begin, *end = json_end;
    ::skip_json_value(begin, end);
    begin = skip_whitespace(begin, end);
    if (begin != end) {
      throw parse::parse_error(begin, "unexpected trailing JSON text");
    }
  }
  catch (const parse::parse_error &e)
  {
    stringstream ss;
    std::string line_prev, line_cur;
    int line, column;
    get_error_line_column(json_begin, json_end, e.get_position(), line_prev, line_cur, line, column);
    ss << "Error validating JSON at line " << line << ", column " << column << "\n";
    ss << "Message: " << e.what() << "\n";
    print_json_parse_error_marker(ss, line_prev, line_cur, line, column);
    throw invalid_argument(ss.str());
  }
}

void dynd::parse_json(nd::array &out, const char *json_begin, const char *json_end, const eval::eval_context *ectx)
{
  try
  {
    const char *begin = json_begin, *end = json_end;
    ndt::type tp = out.get_type();
    ::parse_json(tp, out.get_arrmeta(), out.get_readwrite_originptr(), begin, end, ectx);
    begin = skip_whitespace(begin, end);
    if (begin != end) {
      throw json_parse_error(begin, "unexpected trailing JSON text", tp);
    }
  }
  catch (const json_parse_error &e)
  {
    stringstream ss;
    std::string line_prev, line_cur;
    int line, column;
    get_error_line_column(json_begin, json_end, e.get_position(), line_prev, line_cur, line, column);
    ss << "Error parsing JSON at line " << line << ", column " << column << "\n";
    ss << "DyND Type: " << e.get_type() << "\n";
    ss << "Message: " << e.what() << "\n";
    print_json_parse_error_marker(ss, line_prev, line_cur, line, column);
    throw invalid_argument(ss.str());
  }
  catch (const parse::parse_error &e)
  {
    stringstream ss;
    std::string line_prev, line_cur;
    int line, column;
    get_error_line_column(json_begin, json_end, e.get_position(), line_prev, line_cur, line, column);
    ss << "Error parsing JSON at line " << line << ", column " << column << "\n";
    ss << "Message: " << e.what() << "\n";
    print_json_parse_error_marker(ss, line_prev, line_cur, line, column);
    throw invalid_argument(ss.str());
  }
}

nd::array dynd::parse_json(const ndt::type &tp, const char *json_begin, const char *json_end,
                           const eval::eval_context *ectx)
{
  nd::array result;
  result = nd::empty(tp);
  parse_json(result, json_begin, json_end, ectx);
  if (!tp.is_builtin()) {
    tp.extended()->arrmeta_finalize_buffers(result.get_arrmeta());
  }
  return result;
}

static ndt::type discover_type(const char *&begin, const char *end)
{
  begin = skip_whitespace(begin, end);
  if (begin == end) {
    throw parse::parse_error(begin, "malformed JSON, expecting an element");
  }
  char c = *begin;
  switch (c) {
  // Object
  case '{': {
    ++begin;
    if (parse_token(begin, end, "}")) {
      return ndt::struct_type::make();
    }
    std::vector<std::string> names;
    std::vector<ndt::type> types;
    for (;;) {
      const char *strbegin, *strend;
      bool escaped;
      begin = skip_whitespace(begin, end);
      if (!parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
        throw parse::parse_error(begin, "expected string for name in object dict");
      }
      names.emplace_back(strbegin, strend - strbegin);
      if (!parse_token(begin, end, ":")) {
        throw parse::parse_error(begin, "expected ':' separating name from value in object dict");
      }
      types.push_back(discover_type(begin, end));
      if (!parse_token(begin, end, ",")) {
        break;
      }
    }
    if (!parse_token(begin, end, "}")) {
      throw parse::parse_error(begin, "expected object separator ',' or terminator '}'");
    }
    return ndt::struct_type::make(names, types);
  }
  // Array
  case '[': {
    ++begin;
    if (parse_token(begin, end, "]")) {
      return ndt::tuple_type::make();
    }
    std::vector<ndt::type> types;
    ndt::type common_tp = discover_type(begin, end);
    types.push_back(common_tp);
    for (;;) {
      if (!parse_token(begin, end, ",")) {
        break;
      }
      const ndt::type &tp = discover_type(begin, end);
      if (!common_tp.is_null()) {
        common_tp = ndt::common_type(common_tp, tp);
      }
      types.push_back(tp);
    }
    if (!parse_token(begin, end, "]")) {
      throw parse::parse_error(begin, "expected array separator ',' or terminator ']'");
    }
    if (common_tp.is_null()) {
      return ndt::tuple_type::make(types);
    }
    return ndt::make_fixed_dim(types.size(), common_tp);
  }
  case '"': {
    const char *strbegin, *strend;
    bool escaped;
    if (!parse::parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
      throw parse::parse_error(begin, "invalid string");
    }
    return ndt::type(string_type_id);
  }
  case 'T':
  case 't':
    ++begin;
    if (parse_token(begin, end, "rue")) {
      return ndt::type::make<bool1>();
    }
    throw parse::parse_error(begin, "invalid json value");
  case 'F':
  case 'f':
    ++begin;
    if (parse_token(begin, end, "alse")) {
      return ndt::type::make<bool1>();
    }
    throw parse::parse_error(begin, "invalid json value");
  case 'n':
    if (parse_token(begin, end, "null")) {
      return ndt::option_type::make(ndt::type("Any"));
    }
    throw parse::parse_error(begin, "invalid json value");
  default:
    if (c == '-' || ('0' <= c && c <= '9')) {
      const char *nbegin = NULL, *nend = NULL;
      if (!parse::parse_json_number_no_ws(begin, end, nbegin, nend)) {
        throw parse::parse_error(begin, "invalid number");
      }
      int64_t int_val;
      if (!parse_int64(int_val, nbegin, nend)) {
        return ndt::type::make<int64>();
      }
      double float_val;
      if (!parse_double(float_val, nbegin, nend)) {
        return ndt::type::make<double>();
      }
      throw parse::parse_error(begin, "invalid json value");
    } else {
      throw parse::parse_error(begin, "invalid json value");
    }
  }

  throw runtime_error("json parsing error");
}

void ndt::json::discover(ndt::type &res, const char *json_begin, const char *json_end)
{
  try
  {
    const char *begin = json_begin, *end = json_end;
    res = ::discover_type(begin, end);
    begin = skip_whitespace(begin, end);
    if (begin != end) {
      throw parse::parse_error(begin, "unexpected trailing JSON text");
    }
  }
  catch (const parse::parse_error &e)
  {
    stringstream ss;
    std::string line_prev, line_cur;
    int line, column;
    get_error_line_column(json_begin, json_end, e.get_position(), line_prev, line_cur, line, column);
    ss << "Error validating JSON at line " << line << ", column " << column << "\n";
    ss << "Message: " << e.what() << "\n";
    print_json_parse_error_marker(ss, line_prev, line_cur, line, column);
    throw invalid_argument(ss.str());
  }
}
