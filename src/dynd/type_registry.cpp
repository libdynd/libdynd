//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type_registry.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

DYNDT_API vector<id_info> &detail::infos() {
  static vector<id_info> infos{{},
                               {"Any", any_kind_id, base_ids<any_kind_id>()},
                               {"Scalar", scalar_kind_id, base_ids<scalar_kind_id>()},
                               {"Bool", bool_kind_id, base_ids<bool_kind_id>()},
                               {"bool", bool_id, base_ids<bool_id>()},
                               {"Int", int_kind_id, base_ids<int_kind_id>()},
                               {"int8", int8_id, base_ids<int8_id>()},
                               {"int16", int16_id, base_ids<int16_id>()},
                               {"int32", int32_id, base_ids<int32_id>()},
                               {"int64", int64_id, base_ids<int64_id>()},
                               {"int128", int128_id, base_ids<int128_id>()},
                               {"UInt", uint_kind_id, base_ids<uint_kind_id>()},
                               {"uint8", uint8_id, base_ids<uint8_id>()},
                               {"uint16", uint16_id, base_ids<uint16_id>()},
                               {"uint32", uint32_id, base_ids<uint32_id>()},
                               {"uint64", uint64_id, base_ids<uint64_id>()},
                               {"uint128", uint128_id, base_ids<uint128_id>()},
                               {"Float", float_kind_id, base_ids<float_kind_id>()},
                               {"float16", float16_id, base_ids<float16_id>()},
                               {"float32", float32_id, base_ids<float32_id>()},
                               {"float64", float64_id, base_ids<float64_id>()},
                               {"float128", float128_id, base_ids<float128_id>()},
                               {"Complex", complex_kind_id, base_ids<complex_kind_id>()},
                               {"complex32", complex_float32_id, base_ids<complex_float32_id>()},
                               {"complex64", complex_float64_id, base_ids<complex_float64_id>()},
                               {"void", void_id, base_ids<void_id>()},
                               {"Dim", dim_kind_id, base_ids<dim_kind_id>()},
                               {"Bytes", bytes_kind_id, base_ids<bytes_kind_id>()},
                               {"FixedBytes", fixed_bytes_kind_id, base_ids<fixed_bytes_kind_id>()},
                               {"fixed_bytes", fixed_bytes_id, base_ids<fixed_bytes_id>()},
                               {"bytes", bytes_id, base_ids<bytes_id>()},
                               {"String", string_kind_id, base_ids<string_kind_id>()},
                               {"FixedString", fixed_string_kind_id, base_ids<fixed_string_kind_id>()},
                               {"fixed_string", fixed_string_id, base_ids<fixed_string_id>()},
                               {"char", char_id, base_ids<char_id>()},
                               {"string", string_id, base_ids<string_id>()},
                               {"tuple", tuple_id, base_ids<tuple_id>()},
                               {"struct", struct_id, base_ids<struct_id>()},
                               {"Fixed", fixed_dim_kind_id, base_ids<fixed_dim_kind_id>()},
                               {"fixed", fixed_dim_id, base_ids<fixed_dim_id>()},
                               {"var", var_dim_id, base_ids<var_dim_id>()},
                               {"categorical", categorical_id, base_ids<categorical_id>()},
                               {"option", option_id, base_ids<option_id>()},
                               {"pointer", pointer_id, base_ids<pointer_id>()},
                               {"memory", memory_id, base_ids<memory_id>()},
                               {"type", type_id, base_ids<type_id>()},
                               {"array", array_id, base_ids<array_id>()},
                               {"callable", callable_id, base_ids<callable_id>()},
                               {"Expr", expr_kind_id, base_ids<expr_kind_id>()},
                               {"adapt", adapt_id, base_ids<adapt_id>()},
                               {"expr", expr_id, base_ids<expr_id>()},
                               {"cuda_host", cuda_host_id, {any_kind_id}},
                               {"cuda_device", cuda_device_id, {any_kind_id}},
                               {"State", state_id, {any_kind_id}},
                               {"", typevar_id, base_ids<typevar_id>()},
                               {"", typevar_dim_id, {any_kind_id}},
                               {"", typevar_constructed_id, {any_kind_id}},
                               {"", pow_dimsym_id, {any_kind_id}},
                               {"", ellipsis_dim_id, {any_kind_id}},
                               {"", dim_fragment_id, {any_kind_id}}};

  return infos;
}

DYNDT_API type_id_t dynd::new_id(const char *name, type_id_t base_id) {
  vector<id_info> &infos = detail::infos();

  type_id_t id = static_cast<type_id_t>(infos.size());

  vector<type_id_t> base_ids{base_id};
  for (type_id_t id : infos[base_id].base_ids) {
    base_ids.push_back(id);
  }

  infos.emplace_back(name, id, base_ids);

  return id;
}
