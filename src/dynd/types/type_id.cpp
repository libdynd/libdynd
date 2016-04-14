//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type_registry.hpp>
#include <dynd/types/type_id.hpp>

using namespace std;
using namespace dynd;

ostream &dynd::operator<<(ostream &o, type_id_t id)
{
  switch (id) {
  case uninitialized_id:
    return o << "uninitialized";
  case any_kind_id:
    return o << "Any";
  case scalar_kind_id:
    return o << "Scalar";
  case bool_kind_id:
    return o << "Bool";
  case bool_id:
    return o << "bool";
  case int_kind_id:
    return o << "Int";
  case int8_id:
    return o << "int8";
  case int16_id:
    return o << "int16";
  case int32_id:
    return o << "int32";
  case int64_id:
    return o << "int64";
  case int128_id:
    return o << "int128";
  case uint_kind_id:
    return o << "UInt";
  case uint8_id:
    return o << "uint8";
  case uint16_id:
    return o << "uint16";
  case uint32_id:
    return o << "uint32";
  case uint64_id:
    return o << "uint64";
  case uint128_id:
    return o << "uint128";
  case float16_id:
    return o << "float16";
  case float32_id:
    return o << "float32";
  case float64_id:
    return o << "float64";
  case float128_id:
    return o << "float128";
  case complex_float32_id:
    return o << "complex_float32";
  case complex_float64_id:
    return o << "complex_float64";
  case void_id:
    return o << "void";
  case pointer_id:
    return o << "pointer";
  case bytes_id:
    return o << "bytes";
  case fixed_bytes_id:
    return o << "fixed_bytes";
  case string_id:
    return o << "string";
  case fixed_string_id:
    return o << "fixed_string";
  case categorical_id:
    return o << "categorical";
  case dim_kind_id:
    return o << "dim_kind";
  case fixed_dim_kind_id:
    return o << "fixed_dim_kind";
  case fixed_dim_id:
    return o << "fixed_dim";
  case var_dim_id:
    return o << "var_dim";
  case struct_id:
    return o << "struct";
  case tuple_id:
    return o << "tuple";
  case option_id:
    return o << "option";
  case adapt_id:
    return o << "adapt";
  case kind_sym_id:
    return o << "kind_sym";
  case int_sym_id:
    return o << "int_sym";
  case expr_id:
    return o << "expr";
  case type_id:
    return o << "type";
  case callable_id:
    return o << "callable";
  case typevar_id:
    return o << "typevar";
  case typevar_constructed_id:
    return o << "typevar_constructed";
  case typevar_dim_id:
    return o << "typevar_dim";
  case ellipsis_dim_id:
    return o << "ellipsis_dim";
  default:
    return o << static_cast<underlying_type_t<type_id_t>>(id);
  }
}
