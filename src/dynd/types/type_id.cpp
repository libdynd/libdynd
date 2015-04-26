//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/type_id.hpp>

using namespace std;
using namespace dynd;

std::ostream &dynd::operator<<(std::ostream &o, type_kind_t kind)
{
  switch (kind) {
  case bool_kind:
    return (o << "Bool");
  case sint_kind:
    return (o << "SInt");
  case uint_kind:
    return (o << "UInt");
  case real_kind:
    return (o << "Real");
  case complex_kind:
    return (o << "Complex");
  case string_kind:
    return (o << "String");
  case bytes_kind:
    return (o << "Bytes");
  case void_kind:
    return (o << "Void");
  case datetime_kind:
    return (o << "Datetime");
  case type_kind:
    return (o << "Type");
  case dim_kind:
    return (o << "Dim");
  case struct_kind:
    return (o << "Struct");
  case tuple_kind:
    return (o << "Tuple");
  case dynamic_kind:
    return (o << "Dynamic");
  case expr_kind:
    return (o << "Expr");
  case option_kind:
    return (o << "Option");
  case memory_kind:
    return (o << "Memory");
  case function_kind:
    return (o << "Function");
  case kind_kind:
    return (o << "Kind");
  case pattern_kind:
    return (o << "Pattern");
  case custom_kind:
    return (o << "Custom");
  default:
    return (o << "(unknown kind " << (int)kind << ")");
  }
}

std::ostream &dynd::operator<<(std::ostream &o, type_id_t tid)
{
  switch (tid) {
  case uninitialized_type_id:
    return (o << "uninitialized");
  case bool_type_id:
    return (o << "bool");
  case int8_type_id:
    return (o << "int8");
  case int16_type_id:
    return (o << "int16");
  case int32_type_id:
    return (o << "int32");
  case int64_type_id:
    return (o << "int64");
  case int128_type_id:
    return (o << "int128");
  case uint8_type_id:
    return (o << "uint8");
  case uint16_type_id:
    return (o << "uint16");
  case uint32_type_id:
    return (o << "uint32");
  case uint64_type_id:
    return (o << "uint64");
  case uint128_type_id:
    return (o << "uint128");
  case float16_type_id:
    return (o << "float16");
  case float32_type_id:
    return (o << "float32");
  case float64_type_id:
    return (o << "float64");
  case float128_type_id:
    return (o << "float128");
  case complex_float32_type_id:
    return (o << "complex_float32");
  case complex_float64_type_id:
    return (o << "complex_float64");
  case void_type_id:
    return (o << "void");
  case void_pointer_type_id:
    return (o << "void_pointer");
  case pointer_type_id:
    return (o << "pointer");
  case bytes_type_id:
    return (o << "bytes");
  case fixed_bytes_type_id:
    return (o << "fixed_bytes");
  case string_type_id:
    return (o << "string");
  case fixed_string_type_id:
    return (o << "fixed_string");
  case categorical_type_id:
    return (o << "categorical");
  case date_type_id:
    return (o << "date");
  case time_type_id:
    return (o << "time");
  case datetime_type_id:
    return (o << "datetime");
  case busdate_type_id:
    return (o << "busdate");
  case json_type_id:
    return (o << "json");
  case fixed_dim_type_id:
    return (o << "fixed_dim");
  case var_dim_type_id:
    return (o << "var_dim");
  case struct_type_id:
    return (o << "struct");
  case tuple_type_id:
    return (o << "tuple");
  case c_contiguous_type_id:
    return (o << "C");
  case option_type_id:
    return (o << "option");
  case ndarrayarg_type_id:
    return (o << "ndarrayarg");
  case kind_sym_type_id:
    return (o << "kind_sym");
  case int_sym_type_id:
    return (o << "int_sym");
  case convert_type_id:
    return (o << "convert");
  case byteswap_type_id:
    return (o << "byteswap");
  case view_type_id:
    return (o << "view");
  case property_type_id:
    return (o << "property");
  case expr_type_id:
    return (o << "expr");
  case unary_expr_type_id:
    return (o << "unary_expr");
  case groupby_type_id:
    return (o << "groupby");
  case type_type_id:
    return (o << "type");
  case arrfunc_type_id:
    return (o << "arrfunc");
  case typevar_type_id:
    return (o << "typevar");
  case typevar_constructed_type_id:
    return (o << "typevar_constructed");
  case typevar_dim_type_id:
    return (o << "typevar_dim");
  case ellipsis_dim_type_id:
    return (o << "ellipsis_dim");
  default:
    return (o << "(unknown type id " << (int)tid << ")");
  }
}
