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
                               {"Any", uninitialized_id, ndt::type(), nullptr, nullptr},
                               {"Scalar", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"Bool", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"bool", bool_kind_id, ndt::type(), nullptr, nullptr},
                               {"Int", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"int8", int_kind_id, ndt::type(), nullptr, nullptr},
                               {"int16", int_kind_id, ndt::type(), nullptr, nullptr},
                               {"int32", int_kind_id, ndt::type(), nullptr, nullptr},
                               {"int64", int_kind_id, ndt::type(), nullptr, nullptr},
                               {"int128", int_kind_id, ndt::type(), nullptr, nullptr},
                               {"UInt", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"uint8", uint_kind_id, ndt::type(), nullptr, nullptr},
                               {"uint16", uint_kind_id, ndt::type(), nullptr, nullptr},
                               {"uint32", uint_kind_id, ndt::type(), nullptr, nullptr},
                               {"uint64", uint_kind_id, ndt::type(), nullptr, nullptr},
                               {"uint128", uint_kind_id, ndt::type(), nullptr, nullptr},
                               {"Float", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"float16", float_kind_id, ndt::type(), nullptr, nullptr},
                               {"float32", float_kind_id, ndt::type(), nullptr, nullptr},
                               {"float64", float_kind_id, ndt::type(), nullptr, nullptr},
                               {"float128", float_kind_id, ndt::type(), nullptr, nullptr},
                               {"Complex", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"complex32", complex_kind_id, ndt::type(), nullptr, nullptr},
                               {"complex64", complex_kind_id, ndt::type(), nullptr, nullptr},
                               {"void", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"Dim", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"Bytes", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"FixedBytes", bytes_kind_id, ndt::type(), nullptr, nullptr},
                               {"fixed_bytes", fixed_bytes_kind_id, ndt::type(), nullptr, nullptr},
                               {"bytes", bytes_kind_id, ndt::type(), nullptr, nullptr},
                               {"String", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"FixedString", string_kind_id, ndt::type(), nullptr, nullptr},
                               {"fixed_string", fixed_string_kind_id, ndt::type(), nullptr, nullptr},
                               {"char", string_kind_id, ndt::type(), nullptr, nullptr},
                               {"string", string_kind_id, ndt::type(), nullptr, nullptr},
                               {"tuple", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"struct", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"Fixed", dim_kind_id, ndt::type(), nullptr, nullptr},
                               {"fixed", fixed_dim_kind_id, ndt::type(), nullptr, nullptr},
                               {"var", dim_kind_id, ndt::type(), nullptr, nullptr},
                               {"categorical", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"option", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"pointer", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"memory", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"type", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"array", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"callable", scalar_kind_id, ndt::type(), nullptr, nullptr},
                               {"Expr", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"adapt", expr_kind_id, ndt::type(), nullptr, nullptr},
                               {"expr", expr_kind_id, ndt::type(), nullptr, nullptr},
                               {"cuda_host", memory_id, ndt::type(), nullptr, nullptr},
                               {"cuda_device", memory_id, ndt::type(), nullptr, nullptr},
                               {"State", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"", any_kind_id, ndt::type(), nullptr, nullptr},
                               {"", any_kind_id, ndt::type(), nullptr, nullptr}};

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

  infos.emplace_back(name, base_id, ndt::type(), nullptr, nullptr);

  return id;
}
