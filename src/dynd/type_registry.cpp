//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type_registry.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/datashape_parser.hpp>
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
                               {"Any", uninitialized_id},
                               {"Scalar", any_kind_id},
                               {"Bool", scalar_kind_id},
                               {"bool", bool_kind_id},
                               {"Int", scalar_kind_id},
                               {"int8", int_kind_id},
                               {"int16", int_kind_id},
                               {"int32", int_kind_id},
                               {"int64", int_kind_id},
                               {"int128", int_kind_id},
                               {"UInt", scalar_kind_id},
                               {"uint8", uint_kind_id},
                               {"uint16", uint_kind_id},
                               {"uint32", uint_kind_id},
                               {"uint64", uint_kind_id},
                               {"uint128", uint_kind_id},
                               {"Float", scalar_kind_id},
                               {"float16", float_kind_id},
                               {"float32", float_kind_id},
                               {"float64", float_kind_id},
                               {"float128", float_kind_id},
                               {"Complex", scalar_kind_id},
                               {"complex32", complex_kind_id},
                               {"complex64", complex_kind_id},
                               {"void", scalar_kind_id},
                               {"Dim", any_kind_id},
                               {"Bytes", scalar_kind_id},
                               {"FixedBytes", bytes_kind_id},
                               {"fixed_bytes", fixed_bytes_kind_id},
                               {"bytes", bytes_kind_id},
                               {"String", scalar_kind_id},
                               {"FixedString", string_kind_id},
                               {"fixed_string", fixed_string_kind_id},
                               {"char", string_kind_id},
                               {"string", string_kind_id},
                               {"tuple", scalar_kind_id},
                               {"struct", scalar_kind_id},
                               {"Fixed", dim_kind_id},
                               {"fixed", fixed_dim_kind_id},
                               {"var", dim_kind_id},
                               {"categorical", scalar_kind_id},
                               {"option", any_kind_id},
                               {"pointer", any_kind_id},
                               {"memory", any_kind_id},
                               {"type", scalar_kind_id},
                               {"array", scalar_kind_id},
                               {"callable", scalar_kind_id},
                               {"Expr", any_kind_id},
                               {"adapt", expr_kind_id},
                               {"expr", expr_kind_id},
                               {"cuda_host", memory_id},
                               {"cuda_device", memory_id},
                               {"State", any_kind_id},
                               {"", any_kind_id},
                               {"", any_kind_id},
                               {"", any_kind_id},
                               {"", any_kind_id},
                               {"", any_kind_id},
                               {"", any_kind_id}};

  return infos;
}

ndt::type default_parse_type_args(type_id_t id, const char *&begin, const char *end,
                                  std::map<std::string, ndt::type> &symtable) {
  ndt::type result;
  const char *saved_begin = begin;
  nd::buffer args = dynd::parse_type_constr_args(begin, end, symtable);
  if (!args.is_null()) {
    const vector<id_info> &infos = detail::infos();
    try {
      result = infos[id].construct_type(id, args);
    } catch (const dynd::dynd_exception &e) {
      throw dynd::internal_datashape_parse_error(saved_begin, e.what());
    } catch (const std::exception &e) {
      throw dynd::internal_datashape_parse_error(saved_begin, e.what());
    }
  }
  return result;
}

DYNDT_API type_id_t dynd::new_id(const char *name, type_id_t base_id) {
  vector<id_info> &infos = detail::infos();

  type_id_t id = static_cast<type_id_t>(infos.size());

  infos.emplace_back(name, base_id);

  return id;
}
