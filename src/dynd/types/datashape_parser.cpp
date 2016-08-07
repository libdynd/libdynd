//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <set>
#include <stdexcept>

#include <dynd/parse_util.hpp>
#include <dynd/type_registry.hpp>
#include <dynd/types/array_type.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/datashape_parser.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/pow_dimsym_type.hpp>
#include <dynd/types/state_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/typevar_constructed_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

// Simple recursive descent parser for a subset of the Blaze datashape grammar.
// (Blaze grammar modified slightly to work this way)

static const map<std::string, ndt::type> &builtin_types() {
  static map<std::string, ndt::type> bit;
  if (bit.empty()) {
    bit["intptr"] = ndt::make_type<intptr_t>();
    bit["uintptr"] = ndt::make_type<uintptr_t>();
    bit["size"] = ndt::make_type<size_t>();
    bit["real"] = ndt::make_type<double>();
    bit["complex64"] = ndt::make_type<dynd::complex<float32>>();
    bit["complex128"] = ndt::make_type<dynd::complex<float64>>();
    bit["complex"] = ndt::make_type<dynd::complex<double>>();
    bit["type"] = ndt::make_type<ndt::type_type>();
  }

  return bit;
}

static bool parse_name_or_number(const char *&rbegin, const char *end, const char *&out_nbegin, const char *&out_nend) {
  const char *begin = rbegin;
  // NAME
  if (parse_name_no_ws(begin, end, out_nbegin, out_nend) ||
      parse_unsigned_int_no_ws(begin, end, out_nbegin, out_nend)) {
    rbegin = begin;
    return true;
  }
  return false;
}

// fixed_type : fixed[N] * rhs_expression
static ndt::type parse_fixed_dim_parameters(const char *&rbegin, const char *end,
                                            map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  if (datashape::parse_token(begin, end, '[')) {
    const char *saved_begin = begin;
    std::string dim_size_str = datashape::parse_number(begin, end);
    if (dim_size_str.empty()) {
      throw datashape::internal_parse_error(saved_begin, "expected dimension size");
    }
    intptr_t dim_size = (intptr_t)std::atoll(dim_size_str.c_str());
    if (!datashape::parse_token(begin, end, ']')) {
      throw datashape::internal_parse_error(begin, "expected closing ']'");
    }
    if (!datashape::parse_token(begin, end, '*')) {
      throw datashape::internal_parse_error(begin, "expected dimension separator '*'");
    }
    ndt::type element_tp = datashape::parse(begin, end, symtable);
    if (element_tp.is_null()) {
      throw datashape::internal_parse_error(begin, "expected element type");
    }
    rbegin = begin;
    return ndt::make_fixed_dim(dim_size, element_tp);
  } else {
    throw datashape::internal_parse_error(begin, "expected opening '['");
  }
}

static ndt::type parse_option_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  if (!datashape::parse_token(begin, end, '[')) {
    throw datashape::internal_parse_error(begin, "expected opening '[' after 'option'");
  }
  ndt::type tp = datashape::parse(begin, end, symtable);
  if (tp.is_null()) {
    throw datashape::internal_parse_error(begin, "expected a data type");
  }
  if (!datashape::parse_token(begin, end, ']')) {
    throw datashape::internal_parse_error(begin, "expected closing ']'");
  }
  // TODO catch errors, convert them to datashape::internal_parse_error so the position is
  // shown
  rbegin = begin;
  return ndt::make_type<ndt::option_type>(tp);
}

// complex_type : complex[float_type]
// This is called after 'complex' is already matched
static ndt::type parse_complex_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  if (datashape::parse_token(begin, end, '[')) {
    const char *saved_begin = begin;
    ndt::type tp = datashape::parse(begin, end, symtable);
    if (tp.is_null()) {
      throw datashape::internal_parse_error(begin, "expected a type parameter");
    }
    if (!datashape::parse_token(begin, end, ']')) {
      throw datashape::internal_parse_error(begin, "expected closing ']'");
    }
    if (tp.get_id() == float32_id) {
      rbegin = begin;
      return ndt::make_type<dynd::complex<float>>();
    } else if (tp.get_id() == float64_id) {
      rbegin = begin;
      return ndt::make_type<dynd::complex<double>>();
    } else {
      throw datashape::internal_parse_error(saved_begin, "unsupported real type for complex numbers");
    }
  } else {
    // Default to complex[double] if no parameters are provided
    return ndt::make_type<dynd::complex<double>>();
  }
}

// cuda_host_type : cuda_host[storage_type]
// This is called after 'cuda_host' is already matched
static ndt::type parse_cuda_host_parameters(const char *&rbegin, const char *end,
                                            map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  if (datashape::parse_token(begin, end, '[')) {
#ifdef DYND_CUDA
    ndt::type tp = datashape::parse(begin, end, symtable);
    if (tp.is_null()) {
      throw datashape::internal_parse_error(begin, "expected a type parameter");
    }
    if (!datashape::parse_token(begin, end, ']')) {
      throw datashape::internal_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::make_cuda_host(tp);
#else
    // Silence the unused parameter warning
    symtable.empty();
    throw datashape::internal_parse_error(begin, "cuda_host type is not available");
#endif // DYND_CUDA
  } else {
    throw datashape::internal_parse_error(begin, "expected opening '['");
  }
}

// cuda_device_type : cuda_device[storage_type]
// This is called after 'cuda_device' is already matched
static ndt::type parse_cuda_device_parameters(const char *&rbegin, const char *end,
                                              map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  if (datashape::parse_token(begin, end, '[')) {
#ifdef DYND_CUDA
    ndt::type tp = datashape::parse(begin, end, symtable);
    if (tp.is_null()) {
      throw datashape::internal_parse_error(begin, "expected a type parameter");
    }
    if (!datashape::parse_token(begin, end, ']')) {
      throw datashape::internal_parse_error(begin, "expected closing ']'");
    }
    rbegin = begin;
    return ndt::make_cuda_device(tp);
#else
    // Silence the unused parameter warning
    symtable.empty();
    throw datashape::internal_parse_error(begin, "cuda_device type is not available");
#endif // DYND_CUDA
  } else {
    throw datashape::internal_parse_error(begin, "expected opening '['");
  }
}

static ndt::type parse_pointer_parameters(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  if (!datashape::parse_token(begin, end, '[')) {
    throw datashape::internal_parse_error(begin, "expected opening '[' after 'pointer'");
  }
  ndt::type tp = datashape::parse(begin, end, symtable);
  if (tp.is_null()) {
    throw datashape::internal_parse_error(begin, "expected a data type");
  }
  if (!datashape::parse_token(begin, end, ']')) {
    throw datashape::internal_parse_error(begin, "expected closing ']'");
  }
  // TODO catch errors, convert them to datashape::internal_parse_error so the position is shown
  rbegin = begin;
  return ndt::make_type<ndt::pointer_type>(tp);
}

// datashape_list : datashape COMMA datashape_list RBRACKET
//                | datashape RBRACKET
static nd::buffer parse_datashape_list(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;

  vector<ndt::type> dlist;
  ndt::type arg = datashape::parse(begin, end, symtable);
  if (arg.is_null()) {
    return nd::buffer();
  }
  dlist.push_back(arg);
  for (;;) {
    if (!datashape::parse_token(begin, end, ',')) {
      if (!datashape::parse_token(begin, end, ']')) {
        return nd::buffer();
      } else {
        break;
      }
    }
    arg = datashape::parse(begin, end, symtable);
    if (arg.is_null()) {
      throw datashape::internal_parse_error(begin, "Expected a dynd type or a terminating ']'");
    }
    dlist.push_back(arg);
  }

  rbegin = begin;
  return dlist;
}

// integer_list : INTEGER COMMA integer_list RBRACKET
//              | INTEGER RBRACKET
static nd::buffer parse_integer_list(const char *&rbegin, const char *end) {
  const char *begin = rbegin;

  vector<int64_t> dlist;
  const char *strbegin, *strend;
  if (!parse_int_no_ws(begin, end, strbegin, strend)) {
    return nd::buffer();
  }
  dlist.push_back(parse<int64_t>(strbegin, strend));
  for (;;) {
    if (!datashape::parse_token(begin, end, ',')) {
      if (datashape::parse_token(begin, end, ']')) {
        rbegin = begin;
        return dlist;
      } else {
        return nd::buffer();
      }
    }
    skip_whitespace_and_pound_comments(begin, end);
    if (!parse_int_no_ws(begin, end, strbegin, strend)) {
      throw datashape::internal_parse_error(begin, "Expected an integer or a terminating ']'");
    }
    dlist.push_back(parse<int64_t>(strbegin, strend));
  }
}

// string_list : STRING COMMA string_list RBRACKET
//             | STRING RBRACKET
static nd::buffer parse_string_list(const char *&rbegin, const char *end) {
  const char *begin = rbegin;

  vector<std::string> dlist;
  std::string str;
  if (!datashape::parse_quoted_string(begin, end, str)) {
    return nd::buffer();
  }
  dlist.push_back(str);
  for (;;) {
    if (!datashape::parse_token(begin, end, ',')) {
      if (!datashape::parse_token(begin, end, ']')) {
        return nd::buffer();
      } else {
        break;
      }
    }
    if (!datashape::parse_quoted_string(begin, end, str)) {
      throw datashape::internal_parse_error(begin, "Expected a string");
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
static nd::buffer parse_type_arg(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;

  skip_whitespace_and_pound_comments(begin, end);

  const char *strbegin, *strend;
  if (parse_int_no_ws(begin, end, strbegin, strend)) {
    rbegin = begin;
    return parse<int64_t>(strbegin, strend);
  }

  std::string str;
  if (datashape::parse_quoted_string(begin, end, str)) {
    rbegin = begin;
    return str;
  }

  if (parse_token(begin, end, '[')) {
    nd::buffer result;
    result = parse_integer_list(begin, end);
    if (result.is_null()) {
      result = parse_string_list(begin, end);
    }
    if (result.is_null()) {
      result = parse_datashape_list(begin, end, symtable);
    }
    if (result.is_null()) {
      result = nd::buffer::empty(ndt::make_fixed_dim(0, ndt::make_type<void>()));
    }
    rbegin = begin;
    return result;
  }

  ndt::type tp = datashape::parse(begin, end, symtable);
  if (!tp.is_null()) {
    rbegin = begin;
    return tp;
  }

  return nd::buffer();
}

/**
 * Limited low-level tuple assignment of default-layout data into an uninitialized destination, moves the values so it
 * can work with data of type "type"
 */
template <class Type>
static void move_to_tuple(const ndt::type &tp, const char *meta, char *data, vector<nd::buffer> &values) {
  // Copy all the positional arguments
  auto types = tp.extended<Type>()->get_field_types_raw();
  auto meta_offsets = tp.extended<Type>()->get_arrmeta_offsets_raw();
  auto data_offsets = reinterpret_cast<const uintptr_t *>(meta);
  for (size_t i = 0; i != values.size(); ++i) {
    const nd::buffer &buf = values[i];
    if (buf.get_type() != types[i] ||
        memcmp(buf.get()->metadata(), meta + meta_offsets[i], types[i].get_arrmeta_size()) != 0) {
      throw runtime_error("internal error concatenating datashape type arguments in dynd datashape parser");
    }
    memcpy(data + data_offsets[i], buf.cdata(), types[i].get_default_data_size());
    // Destroy the nd::buffer without destructing its data
    nd::buffer tmp;
    values[i].swap(tmp);
    if (!tmp.get_type().is_builtin()) {
      tmp.get_type().get()->arrmeta_destruct(tmp.get()->metadata());
      const_cast<ndt::type &>(tmp.get_type()) = ndt::type();
    }
  }
}

/**
 * Special-cased nd::buffer concatenation function, tailored for arg buffers. Ultimately, some basic form of nd::buffer
 * concatenation needs to make its way into dyndt, likely via the basic nd::callable primitives.
 */
static nd::buffer move_concatenate_arg_buffers(vector<nd::buffer> &pos_args, const vector<std::string> &kw_names,
                                               vector<nd::buffer> &kw_args) {
  // Create type "((type0, ...), {kw0: kwtype0, ...})"
  vector<ndt::type> pos_arg_types;
  transform(pos_args.begin(), pos_args.end(), back_inserter(pos_arg_types),
            [](const nd::buffer &a) { return a.get_type(); });

  vector<ndt::type> kw_arg_types;
  transform(kw_args.begin(), kw_args.end(), back_inserter(kw_arg_types),
            [](const nd::buffer &a) { return a.get_type(); });

  ndt::type result_tp = ndt::make_type<ndt::tuple_type>(
      {ndt::make_type<ndt::tuple_type>(pos_arg_types), ndt::make_type<ndt::struct_type>(kw_names, kw_arg_types)});

  nd::buffer result = nd::buffer::empty(result_tp);

  auto outer_types = result.get_type().extended<ndt::tuple_type>()->get_field_types_raw();
  auto outer_meta_offsets = result.get_type().extended<ndt::tuple_type>()->get_arrmeta_offsets_raw();
  auto outer_data_offsets = reinterpret_cast<const uintptr_t *>(result.get()->metadata());

  move_to_tuple<ndt::tuple_type>(outer_types[0], result.get()->metadata() + outer_meta_offsets[0],
                                 result.data() + outer_data_offsets[0], pos_args);
  move_to_tuple<ndt::struct_type>(outer_types[1], result.get()->metadata() + outer_meta_offsets[1],
                                  result.data() + outer_data_offsets[1], kw_args);

  return result;
}

// type_arg_list : type_arg COMMA type_arg_list
//               | type_kwarg_list
//               | type_arg
// type_kwarg_list : type_kwarg COMMA type_kwarg_list
//                 | type_kwarg
// type_kwarg : NAME_LOWER EQUAL type_arg
// type_constr_args : LBRACKET type_arg_list RBRACKET
nd::buffer datashape::parse_type_constr_args(const char *&rbegin, const char *end,
                                             map<std::string, ndt::type> &symtable) {
  nd::buffer result;

  const char *begin = rbegin;
  if (!datashape::parse_token(begin, end, '[')) {
    // Return an empty array if there is no leading bracket
    return result;
  }

  if (datashape::parse_token(begin, end, ']')) {
    return nd::buffer::empty(
        ndt::make_type<ndt::tuple_type>({ndt::make_type<ndt::tuple_type>(), ndt::make_type<ndt::struct_type>()}));
  }

  vector<nd::buffer> pos_args;
  vector<nd::buffer> kw_args;
  vector<std::string> kw_names;

  const char *field_name_begin, *field_name_end;
  bool done = false;

  // First parse all the positional arguments
  for (;;) {
    skip_whitespace_and_pound_comments(begin, end);
    // Look ahead to see the ']' or whether there's a keyword argument
    const char *saved_begin = begin;
    if (parse_name_no_ws(begin, end, field_name_begin, field_name_end) && datashape::parse_token(begin, end, ':')) {
      begin = saved_begin;
      break;
    }
    begin = saved_begin;
    // Parse one positional argument
    nd::buffer arg = parse_type_arg(begin, end, symtable);
    if (arg.is_null()) {
      throw datashape::internal_parse_error(saved_begin, "Expected a positional or keyword type argument");
    }
    pos_args.push_back(arg);
    if (!datashape::parse_token(begin, end, ',')) {
      if (datashape::parse_token(begin, end, ']')) {
        done = true;
      }
      break;
    }
  }

  // Now parse all the keyword arguments
  if (!done) {
    for (;;) {
      const char *saved_begin = begin;
      skip_whitespace_and_pound_comments(begin, end);
      if (!parse_name_no_ws(begin, end, field_name_begin, field_name_end)) {
        throw datashape::internal_parse_error(saved_begin, "Expected a keyword name or terminating ']'");
      }
      if (!datashape::parse_token(begin, end, ':')) {
        throw datashape::internal_parse_error(begin, "Expected ':' between keyword name and parameter");
      }
      nd::buffer arg = parse_type_arg(begin, end, symtable);
      if (arg.is_null()) {
        throw datashape::internal_parse_error(begin, "Expected keyword argument value");
      }
      kw_args.push_back(arg);
      kw_names.push_back(std::string(field_name_begin, field_name_end));
      if (!datashape::parse_token(begin, end, ',')) {
        if (!datashape::parse_token(begin, end, ']')) {
          throw datashape::internal_parse_error(begin, "Expected a ',' or ']'");
        } else {
          break;
        }
      }
    }
  }

  result = move_concatenate_arg_buffers(pos_args, kw_names, kw_args);

  rbegin = begin;
  return result;
}

// record_item_bare : BARENAME COLON rhs_expression
static bool parse_struct_item_bare(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable,
                                   std::string &out_field_name, ndt::type &out_field_type) {
  const char *begin = rbegin;
  const char *field_name_begin, *field_name_end;
  skip_whitespace_and_pound_comments(begin, end);
  if (parse_name_no_ws(begin, end, field_name_begin, field_name_end)) {
    // We successfully parsed a name with no whitespace
    // We don't need to do anything else, because field_name_begin
  } else {
    // This struct item cannot be parsed. Ergo, we return false for failure.
    return false;
  }
  if (!datashape::parse_token(begin, end, ':')) {
    throw datashape::internal_parse_error(begin, "expected ':' after record item name");
  }
  out_field_type = datashape::parse(begin, end, symtable);
  if (out_field_type.is_null()) {
    throw datashape::internal_parse_error(begin, "expected a data type");
  }

  out_field_name.assign(field_name_begin, field_name_end);
  rbegin = begin;
  return true;
}

// struct_item_general : struct_item_bare |
//                       QUOTEDNAME COLON rhs_expression
static bool parse_struct_item_general(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable,
                                      std::string &out_field_name, ndt::type &out_field_type) {
  const char *begin = rbegin;
  const char *field_name_begin, *field_name_end;
  // quoted_out_val and quoted_name are used to hold the field name and to
  // denote if the data given
  //  to this function needed special handling due to quoting of the struct
  //  field names.
  std::string quoted_out_val;
  bool quoted_name = false;
  skip_whitespace_and_pound_comments(begin, end);
  if (parse_name_no_ws(begin, end, field_name_begin, field_name_end)) {
    // We successfully parsed a name with no whitespace
    // We don't need to do anything else, because field_name_begin
  } else if (datashape::parse_quoted_string(begin, end, quoted_out_val)) {
    // datashape::parse_quoted_string must return a new string for us to use because it
    // will parse
    //  and potentially replace things in the string (like escaped characters)
    // It will also remove the surrounding quotes.
    quoted_name = true;
  } else {
    // This struct item cannot be parsed. Ergo, we return false for failure.
    return false;
  }
  if (!datashape::parse_token(begin, end, ':')) {
    throw datashape::internal_parse_error(begin, "expected ':' after record item name");
  }
  out_field_type = datashape::parse(begin, end, symtable);
  if (out_field_type.is_null()) {
    throw datashape::internal_parse_error(begin, "expected a data type");
  }

  if (!quoted_name) {
    // A name that isn't quoted is probably the common case
    out_field_name.assign(field_name_begin, field_name_end);
  } else {
    // If a field name was quoted, datashape::parse_quoted_string() will have parsed and
    // un/re-escaped everything and returned a new string
    // The Return of the String is why we have two different
    // out_field_name.assign() cases
    out_field_name.assign(quoted_out_val);
  }
  rbegin = begin;
  return true;
}

// struct : LBRACE record_item record_item* RBRACE
static ndt::type parse_struct(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  vector<std::string> field_name_list;
  vector<ndt::type> field_type_list;
  std::string field_name;
  ndt::type field_type;
  bool variadic = false;

  if (!datashape::parse_token(begin, end, '{')) {
    return ndt::type();
  }
  if (datashape::parse_token(begin, end, '}')) {
    // Empty struct
    rbegin = begin;
    return ndt::make_type<ndt::struct_type>();
  }
  for (;;) {
    if (datashape::parse_token(begin, end, "...")) {
      if (!datashape::parse_token(begin, end, '}')) {
        throw datashape::internal_parse_error(begin, "expected '}'");
      }
      variadic = true;
      break;
    }

    const char *saved_begin = begin;
    skip_whitespace_and_pound_comments(begin, end);
    if (parse_struct_item_general(begin, end, symtable, field_name, field_type)) {
      field_name_list.push_back(field_name);
      field_type_list.push_back(field_type);
    } else {
      throw datashape::internal_parse_error(saved_begin, "expected a record item");
    }

    if (datashape::parse_token(begin, end, ',')) {
      if (!field_name_list.empty() && datashape::parse_token(begin, end, '}')) {
        break;
      }
    } else if (datashape::parse_token(begin, end, '}')) {
      break;
    } else {
      throw datashape::internal_parse_error(begin, "expected ',' or '}'");
    }
  }

  rbegin = begin;
  return ndt::make_type<ndt::struct_type>(field_name_list, field_type_list, variadic);
}

// funcproto_kwds : record_item, record_item*
static ndt::type parse_funcproto_kwds(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  vector<std::string> field_name_list;
  vector<ndt::type> field_type_list;
  std::string field_name;
  ndt::type field_type;
  bool variadic = false;

  for (;;) {
    // Check for variadic ending
    if (datashape::parse_token(begin, end, "...")) {
      if (!datashape::parse_token(begin, end, ')')) {
        throw datashape::internal_parse_error(begin, "expected ',' or ')' in callable prototype");
      }
      variadic = true;
      break;
    }

    const char *saved_begin = begin;
    skip_whitespace_and_pound_comments(begin, end);
    if (parse_struct_item_bare(begin, end, symtable, field_name, field_type)) {
      field_name_list.push_back(field_name);
      field_type_list.push_back(field_type);
    } else {
      throw datashape::internal_parse_error(saved_begin, "expected a kwd arg in callable prototype");
    }

    if (datashape::parse_token(begin, end, ',')) {
      if (!field_name_list.empty() && datashape::parse_token(begin, end, ')')) {
        break;
      }
    } else if (datashape::parse_token(begin, end, ')')) {
      break;
    } else {
      throw datashape::internal_parse_error(begin, "expected ',' or ')' in callable prototype");
    }
  }

  rbegin = begin;
  return ndt::make_type<ndt::struct_type>(field_name_list, field_type_list, variadic);
}

// tuple : LPAREN tuple_item tuple_item* RPAREN
// funcproto : tuple -> type
static ndt::type parse_tuple_or_funcproto(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  vector<ndt::type> field_type_list;
  bool variadic = false;

  if (!datashape::parse_token(begin, end, '(')) {
    return ndt::type();
  }
  if (!datashape::parse_token(begin, end, ')')) {
    for (;;) {
      ndt::type tp;
      // Look ahead to see if we've got "BARENAME:" or "..., BARENAME:" coming
      // next, and if so, parse the keyword arguments and return.
      const char *saved_begin = begin, *kwds_begin = begin;
      const char *field_name_begin, *field_name_end;
      if (datashape::parse_token(begin, end, "...") && datashape::parse_token(begin, end, ',')) {
        variadic = true;
        kwds_begin = begin;
      }
      skip_whitespace_and_pound_comments(begin, end);
      if (parse_name_no_ws(begin, end, field_name_begin, field_name_end)) {
        if (datashape::parse_token(begin, end, ':')) {
          // process the keyword arguments
          ndt::type funcproto_kwd;
          begin = kwds_begin;
          funcproto_kwd = parse_funcproto_kwds(begin, end, symtable);
          if (!funcproto_kwd.is_null()) {
            if (!datashape::parse_token(begin, end, "->")) {
              rbegin = begin;
              return ndt::make_type<ndt::tuple_type>(field_type_list.size(), field_type_list.data());
            }

            ndt::type return_type = datashape::parse(begin, end, symtable);
            if (return_type.is_null()) {
              throw datashape::internal_parse_error(begin, "expected function prototype return type");
            }
            rbegin = begin;
            return ndt::make_type<ndt::callable_type>(
                return_type, ndt::make_type<ndt::tuple_type>(field_type_list.size(), field_type_list.data(), variadic),
                funcproto_kwd);
          } else {
            throw datashape::internal_parse_error(begin, "expected funcproto keyword arguments");
          }
        }
      }
      begin = saved_begin;

      // Look ahead again to see if the tuple ends with "...)", in which case
      // it's a variadic tuple.
      if (datashape::parse_token(begin, end, "...")) {
        if (datashape::parse_token(begin, end, ')')) {
          variadic = true;
          break;
        }
      }
      begin = saved_begin;

      tp = datashape::parse(begin, end, symtable);

      if (tp.get_id() != uninitialized_id) {
        field_type_list.push_back(tp);
      } else {
        throw datashape::internal_parse_error(begin, "expected a type");
      }

      if (datashape::parse_token(begin, end, ',')) {
        if (!field_type_list.empty() && datashape::parse_token(begin, end, ')')) {
          break;
        }
      } else if (datashape::parse_token(begin, end, ')')) {
        break;
      } else {
        throw datashape::internal_parse_error(begin, "expected ',' or ')'");
      }
    }
  }

  // It might be a function prototype, check for the "->" token
  if (!datashape::parse_token(begin, end, "->")) {
    rbegin = begin;
    return ndt::make_type<ndt::tuple_type>(field_type_list.size(), field_type_list.data(), variadic);
  }

  ndt::type return_type = datashape::parse(begin, end, symtable);
  if (return_type.is_null()) {
    throw datashape::internal_parse_error(begin, "expected function prototype return type");
  }
  rbegin = begin;
  // TODO: I suspect because of the change away from immutable default
  // construction, and
  //       the requirement that arrays into callable constructors are
  //       immutable, that too
  //       many copies may be occurring.
  return ndt::make_type<ndt::callable_type>(
      return_type, ndt::make_type<ndt::tuple_type>(field_type_list.size(), field_type_list.data(), variadic),
      ndt::make_type<ndt::struct_type>(variadic));
}

//    datashape_nooption : dim ASTERISK datashape
//                       | dtype
static ndt::type parse_datashape_nooption(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  ndt::type result;
  const char *begin = rbegin;
  skip_whitespace_and_pound_comments(begin, end);
  // First try "dim ASTERISK ASTERISK datashape", then "dim ASTERISK datashape"
  const char *nbegin, *nend;
  if (parse_name_or_number(begin, end, nbegin, nend)) {
    if (datashape::parse_token(begin, end, "**")) {
      // Save the extents of the base dim token
      const char *bbegin = nbegin;
      const char *bend = nend;
      if (parse_name_or_number(begin, end, nbegin, nend)) {
        if ('1' <= *nbegin && *nbegin <= '9') {
          intptr_t exponent = parse<intptr_t>(nbegin, nend);
          if (!datashape::parse_token(begin, end, '*')) {
            throw datashape::internal_parse_error(begin, "expected a '*' after dimensional power");
          }
          ndt::type element_tp = datashape::parse(begin, end, symtable);
          if ('0' <= *bbegin && *bbegin <= '9') {
            intptr_t dim_size = parse<intptr_t>(bbegin, bend);
            result = ndt::pow(ndt::make_fixed_dim(dim_size, element_tp), exponent);
          } else if (compare_range_to_literal(bbegin, bend, "var")) {
            result = ndt::pow(ndt::make_type<ndt::var_dim_type>(element_tp), exponent);
          } else if (compare_range_to_literal(bbegin, bend, "Fixed")) {
            result = ndt::pow(ndt::make_type<ndt::fixed_dim_kind_type>(element_tp), exponent);
          } else if (isupper(*bbegin)) {
            result = ndt::pow(ndt::make_type<ndt::typevar_dim_type>(std::string(bbegin, bend), element_tp), exponent);
          } else {
            throw datashape::internal_parse_error(bbegin, "invalid dimension type for base of dimensional power");
          }
        } else if (isupper(*nbegin)) {
          std::string exponent_name(nbegin, nend);
          if (datashape::parse_token(begin, end, '*')) {
            if ('0' <= *bbegin && *bbegin <= '9') {
              intptr_t dim_size = parse<intptr_t>(bbegin, bend);
              result = ndt::make_type<ndt::pow_dimsym_type>(ndt::make_fixed_dim(dim_size, ndt::make_type<void>()),
                                                            exponent_name, datashape::parse(begin, end, symtable));
            } else if (compare_range_to_literal(bbegin, bend, "var")) {
              result = ndt::make_type<ndt::pow_dimsym_type>(ndt::make_type<ndt::var_dim_type>(ndt::make_type<void>()),
                                                            exponent_name, datashape::parse(begin, end, symtable));
            } else if (compare_range_to_literal(bbegin, bend, "Fixed")) {
              result =
                  ndt::make_type<ndt::pow_dimsym_type>(ndt::make_type<ndt::fixed_dim_kind_type>(ndt::make_type<void>()),
                                                       exponent_name, datashape::parse(begin, end, symtable));
            } else if (isupper(*bbegin)) {
              result = ndt::make_type<ndt::pow_dimsym_type>(
                  ndt::make_type<ndt::typevar_dim_type>(std::string(bbegin, bend), ndt::make_type<void>()),
                  exponent_name, datashape::parse(begin, end, symtable));
            } else {
              throw datashape::internal_parse_error(bbegin, "invalid dimension type for base of dimensional power");
            }
          }
        } else {
          throw datashape::internal_parse_error(begin, "expected a number or a typevar symbol");
        }
      } else {
        throw datashape::internal_parse_error(begin, "expected a number or a typevar symbol");
      }
    } else if (datashape::parse_token(begin, end, '*')) {
      ndt::type element_tp = datashape::parse(begin, end, symtable);
      if (element_tp.is_null()) {
        throw datashape::internal_parse_error(begin, "expected a dynd type");
      }
      // No type constructor args, just a dim type
      if ('0' <= *nbegin && *nbegin <= '9') {
        intptr_t size = parse<intptr_t>(nbegin, nend);
        result = ndt::make_fixed_dim(size, element_tp);
      } else if (compare_range_to_literal(nbegin, nend, "var")) {
        result = ndt::make_type<ndt::var_dim_type>(element_tp);
      } else if (compare_range_to_literal(nbegin, nend, "Fixed")) {
        result = ndt::make_type<ndt::fixed_dim_kind_type>(element_tp);
      } else if (isupper(*nbegin)) {
        result = ndt::make_type<ndt::typevar_dim_type>(std::string(nbegin, nend), element_tp);
      } else {
        skip_whitespace_and_pound_comments(rbegin, end);
        throw datashape::internal_parse_error(rbegin, "unrecognized dimension type");
      }
    } else if (datashape::parse_token(begin, end, "...")) { // ELLIPSIS
      // A named ellipsis dim
      if (datashape::parse_token(begin, end, '*')) { // ASTERISK
        // An unnamed ellipsis dim
        ndt::type element_tp = datashape::parse(begin, end, symtable);
        if (element_tp.is_null()) {
          throw datashape::internal_parse_error(begin, "expected a dynd type");
        }
        result = ndt::make_type<ndt::ellipsis_dim_type>(std::string(nbegin, nend), element_tp);
      } else {
        throw datashape::internal_parse_error(begin, "expected a '*'");
      }
    } else if (compare_range_to_literal(nbegin, nend, "complex")) {
      result = parse_complex_parameters(begin, end, symtable);
    } else if (compare_range_to_literal(nbegin, nend, "pointer")) {
      result = parse_pointer_parameters(begin, end, symtable);
    } else if (compare_range_to_literal(nbegin, nend, "array")) {
      result = ndt::make_type<ndt::array_type>();
    } else if (compare_range_to_literal(nbegin, nend, "cuda_host")) {
      result = parse_cuda_host_parameters(begin, end, symtable);
    } else if (compare_range_to_literal(nbegin, nend, "cuda_device")) {
      result = parse_cuda_device_parameters(begin, end, symtable);
    } else if (compare_range_to_literal(nbegin, nend, "fixed")) {
      result = parse_fixed_dim_parameters(begin, end, symtable);
    } else if (compare_range_to_literal(nbegin, nend, "option")) {
      result = parse_option_parameters(begin, end, symtable);
    } else if (compare_range_to_literal(nbegin, nend, "State")) {
      result = ndt::make_type<ndt::state_type>();
    } else if (compare_range_to_literal(nbegin, nend, "Categorical")) {
      result = ndt::make_type<ndt::categorical_kind_type>();
    } else {
      std::string n(nbegin, nend);
      const map<std::string, ndt::type> &bit = builtin_types();
      map<std::string, ndt::type>::const_iterator i = bit.find(n);
      if (i != bit.end()) {
        result = i->second;
      } else {
        auto ii = lookup_id_by_name(n);
        if (ii.first != uninitialized_id) {
          skip_whitespace_and_pound_comments(begin, end);
          // Peek ahead for the '[' to determine whether arguments are there to parse
          if (begin > end && *begin == '[') {
            if (ii.second->construct_type == nullptr) {
              skip_whitespace_and_pound_comments(rbegin, end);
              throw datashape::internal_parse_error(rbegin, "this data type does not accept arguments");
            } else if (ii.second->parse_type_args != nullptr) {
              result = ii.second->parse_type_args(ii.first, begin, end, symtable);
            } else {
              // Both `construct_type` and `parse_type_args` must be there or not, raise an error for types where this
              // is not the case.
              skip_whitespace_and_pound_comments(rbegin, end);
              throw datashape::internal_parse_error(rbegin, "internal type registry for this type is inconsistent");
            }
          } else if (!ii.second->singleton_type.is_null()) {
            result = ii.second->singleton_type;
          } else {
            skip_whitespace_and_pound_comments(rbegin, end);
            throw datashape::internal_parse_error(rbegin, "this data type requires arguments");
          }
        } else if (isupper(*nbegin)) {
          if (!datashape::parse_token(begin, end, '[')) {
            result = ndt::make_type<ndt::typevar_type>(std::string(nbegin, nend));
          } else {
            ndt::type arg_tp = datashape::parse(begin, end, symtable);
            if (arg_tp.is_null()) {
              throw datashape::internal_parse_error(begin, "expected a dynd type");
            }
            if (!datashape::parse_token(begin, end, ']')) {
              throw datashape::internal_parse_error(begin, "expected closing ']'");
            }
            result = ndt::make_type<ndt::typevar_constructed_type>(std::string(nbegin, nend), arg_tp);
          }
        } else {
          skip_whitespace_and_pound_comments(rbegin, end);
          throw datashape::internal_parse_error(rbegin, "unrecognized data type");
        }
      }
    }
  } else if (parse_token_no_ws(begin, end, "...")) {
    // An unnamed ellipsis dim
    if (datashape::parse_token(begin, end, '*')) { // ASTERISK
      ndt::type element_type = datashape::parse(begin, end, symtable);
      if (element_type.is_null()) {
        throw datashape::internal_parse_error(begin, "expected a dynd type");
      }
      result = ndt::make_type<ndt::ellipsis_dim_type>(element_type);
    } else {
      throw datashape::internal_parse_error(begin, "expected a '*'");
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
ndt::type datashape::parse(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  skip_whitespace_and_pound_comments(begin, end);
  if (parse_token_no_ws(begin, end, '?')) {
    ndt::type val_tp = parse_datashape_nooption(begin, end, symtable);
    if (!val_tp.is_null()) {
      rbegin = begin;
      return ndt::make_type<ndt::option_type>(val_tp);
    } else {
      return ndt::type();
    }
  } else {
    return parse_datashape_nooption(rbegin, end, symtable);
  }
}

static ndt::type parse_stmt(const char *&rbegin, const char *end, map<std::string, ndt::type> &symtable) {
  const char *begin = rbegin;
  // stmt : TYPE name EQUALS rhs_expression
  // NOTE that this doesn't support parameterized lhs_expression, this is subset
  // of Blaze datashape
  if (datashape::parse_token(begin, end, "type")) {
    const map<std::string, ndt::type> &bit = builtin_types();
    const char *saved_begin = begin;
    const char *tname_begin, *tname_end;
    if (!skip_required_whitespace(begin, end)) {
      if (begin == end) {
        // If it's only "type" by itself, return the "type" type
        rbegin = begin;
        return bit.find("type")->second;
      } else {
        return ndt::type();
      }
    }
    if (!parse_name_no_ws(begin, end, tname_begin, tname_end)) {
      skip_whitespace_and_pound_comments(begin, end);
      if (begin == end) {
        // If it's only "type" by itself, return the "type" type
        rbegin = begin;
        return bit.find("type")->second;
      } else {
        throw datashape::internal_parse_error(saved_begin, "expected an identifier for a type name");
      }
    }
    if (!datashape::parse_token(begin, end, '=')) {
      throw datashape::internal_parse_error(begin, "expected an '='");
    }
    ndt::type result = datashape::parse(begin, end, symtable);
    if (result.is_null()) {
      throw datashape::internal_parse_error(begin, "expected a data type");
    }
    std::string tname(tname_begin, tname_end);
    // ACTION: Put the parsed type in the symbol table
    if (bit.find(tname) != bit.end()) {
      skip_whitespace_and_pound_comments(saved_begin, end);
      throw datashape::internal_parse_error(saved_begin, "cannot redefine a builtin type");
    }
    if (symtable.find(tname) != symtable.end()) {
      skip_whitespace_and_pound_comments(saved_begin, end);
      throw datashape::internal_parse_error(saved_begin, "type name already defined in datashape string");
    }
    symtable[tname] = result;
    rbegin = begin;
    return result;
  } else {
    // stmt : rhs_expression
    return datashape::parse(rbegin, end, symtable);
  }
}

// top : stmt stmt*
static ndt::type parse_top(const char *&begin, const char *end, map<std::string, ndt::type> &symtable) {
  ndt::type result = parse_stmt(begin, end, symtable);
  if (result.is_null()) {
    throw datashape::internal_parse_error(begin, "expected a datashape statement");
  }
  for (;;) {
    ndt::type next = parse_stmt(begin, end, symtable);
    if (next.is_null()) {
      skip_whitespace_and_pound_comments(begin, end);
      if (begin != end) {
        throw datashape::internal_parse_error(begin, "unexpected token in datashape");
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
                                  std::string &out_line_cur, int &out_line, int &out_column) {
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

ndt::type dynd::type_from_datashape(const char *datashape_begin, const char *datashape_end) {
  try {
    // Symbol table for intermediate types declared in the datashape
    map<std::string, ndt::type> symtable;
    // Parse the datashape and construct the type
    const char *begin = datashape_begin, *end = datashape_end;
    return parse_top(begin, end, symtable);
  } catch (const datashape::internal_parse_error &e) {
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

nd::buffer datashape::parse_type_constr_args(const std::string &str) {
  nd::buffer result;
  std::map<std::string, ndt::type> symtable;
  if (!str.empty()) {
    const char *begin = &str[0], *end = &str[0] + str.size();
    try {
      result = parse_type_constr_args(begin, end, symtable);
    } catch (const datashape::internal_parse_error &e) {
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
