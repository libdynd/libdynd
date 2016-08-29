//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/parse_util.hpp>
#include <dynd/type_registry.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/array_type.hpp>
#include <dynd/types/bool_kind_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/complex_kind_type.hpp>
#include <dynd/types/cuda_device_type.hpp>
#include <dynd/types/cuda_host_type.hpp>
#include <dynd/types/datashape_parser.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/float_kind_type.hpp>
#include <dynd/types/int_kind_type.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/uint_kind_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

DYNDT_API vector<id_info> &detail::infos() {
  static vector<id_info> infos{
      {},
      {"Any", uninitialized_id, ndt::make_type<ndt::any_kind_type>(), nullptr, nullptr},
      {"Scalar", any_kind_id, ndt::make_type<ndt::scalar_kind_type>(), nullptr, nullptr},
      {"Bool", scalar_kind_id, ndt::make_type<ndt::bool_kind_type>(), nullptr, nullptr},
      {"bool", bool_kind_id, ndt::make_type<bool>(), nullptr, nullptr},
      {"Int", scalar_kind_id, ndt::make_type<ndt::int_kind_type>(), nullptr, nullptr},
      {"int8", int_kind_id, ndt::make_type<int8_t>(), nullptr, nullptr},
      {"int16", int_kind_id, ndt::make_type<int16_t>(), nullptr, nullptr},
      {"int32", int_kind_id, ndt::make_type<int32_t>(), nullptr, nullptr},
      {"int64", int_kind_id, ndt::make_type<int64_t>(), nullptr, nullptr},
      {"int128", int_kind_id, ndt::make_type<dynd::int128>(), nullptr, nullptr},
      {"UInt", scalar_kind_id, ndt::make_type<ndt::uint_kind_type>(), nullptr, nullptr},
      {"uint8", uint_kind_id, ndt::make_type<uint8_t>(), nullptr, nullptr},
      {"uint16", uint_kind_id, ndt::make_type<uint16_t>(), nullptr, nullptr},
      {"uint32", uint_kind_id, ndt::make_type<uint32_t>(), nullptr, nullptr},
      {"uint64", uint_kind_id, ndt::make_type<uint64_t>(), nullptr, nullptr},
      {"uint128", uint_kind_id, ndt::make_type<dynd::uint128>(), nullptr, nullptr},
      {"Float", scalar_kind_id, ndt::make_type<ndt::float_kind_type>(), nullptr, nullptr},
      {"float16", float_kind_id, ndt::make_type<dynd::float16>(), nullptr, nullptr},
      {"float32", float_kind_id, ndt::make_type<float>(), nullptr, nullptr},
      {"float64", float_kind_id, ndt::make_type<double>(), nullptr, nullptr},
      {"float128", float_kind_id, ndt::make_type<dynd::float128>(), nullptr, nullptr},
      {"Complex", scalar_kind_id, ndt::make_type<ndt::complex_kind_type>(), nullptr, nullptr},
      {"complex32", complex_kind_id, ndt::make_type<complex<float>>(), nullptr, nullptr},
      {"complex64", complex_kind_id, ndt::make_type<complex<double>>(), nullptr, nullptr},
      {"void", scalar_kind_id, ndt::make_type<void>(), nullptr, nullptr},
      {"Dim", any_kind_id, ndt::type(), &ndt::dim_kind_type::construct_type, nullptr},
      {"Bytes", scalar_kind_id, ndt::make_type<ndt::bytes_kind_type>(), nullptr, nullptr},
      {"FixedBytes", bytes_kind_id, ndt::make_type<ndt::fixed_bytes_kind_type>(), nullptr, nullptr},
      {"fixed_bytes", fixed_bytes_kind_id, ndt::type(), nullptr, &ndt::fixed_bytes_type::parse_type_args},
      {"bytes", bytes_kind_id, ndt::make_type<ndt::bytes_type>(1), nullptr, &ndt::bytes_type::parse_type_args},
      {"String", scalar_kind_id, ndt::make_type<ndt::string_kind_type>(), nullptr, nullptr},
      {"FixedString", string_kind_id, ndt::make_type<ndt::fixed_string_kind_type>(), nullptr, nullptr},
      {"fixed_string", fixed_string_kind_id, ndt::type(), nullptr, &ndt::fixed_string_type::parse_type_args},
      {"char", string_kind_id, ndt::make_type<ndt::char_type>(), nullptr, &ndt::char_type::parse_type_args},
      {"string", string_kind_id, ndt::make_type<dynd::string>(), nullptr, nullptr},
      {"tuple", scalar_kind_id, ndt::type(), nullptr, nullptr},
      {"struct", scalar_kind_id, ndt::type(), nullptr, nullptr},
      {"Fixed", dim_kind_id, ndt::type(), &ndt::fixed_dim_kind_type::construct_type, nullptr},
      {"fixed", fixed_dim_kind_id, ndt::type(), nullptr, &ndt::fixed_dim_type::parse_type_args},
      {"var", dim_kind_id, ndt::type(), &ndt::var_dim_type::construct_type, nullptr},
      {"Categorical", scalar_kind_id, ndt::make_type<ndt::categorical_kind_type>(), nullptr, nullptr},
      {"categorical", categorical_kind_id, ndt::type(), nullptr, nullptr /*&ndt::categorical_type::parse_type_args*/},
      {"option", any_kind_id, ndt::type(), nullptr, &ndt::option_type::parse_type_args},
      {"pointer", any_kind_id, ndt::type(), nullptr, &ndt::pointer_type::parse_type_args},
      {"memory", any_kind_id, ndt::type(), nullptr, nullptr},
      {"type", scalar_kind_id, ndt::make_type<ndt::type_type>(), nullptr, nullptr},
      {"array", scalar_kind_id, ndt::make_type<ndt::array_type>(), nullptr, nullptr},
      {"callable", scalar_kind_id, ndt::type(), nullptr, nullptr},
      {"Expr", any_kind_id, ndt::type(), nullptr, nullptr},
      {"adapt", expr_kind_id, ndt::type(), nullptr, nullptr},
      {"expr", expr_kind_id, ndt::type(), nullptr, nullptr},
      {"cuda_host", memory_id, ndt::type(), nullptr, &ndt::cuda_host_type::parse_type_args},
      {"cuda_device", memory_id, ndt::type(), nullptr, &ndt::cuda_device_type::parse_type_args},
      {"State", any_kind_id, ndt::type(), nullptr, nullptr},
      {"", any_kind_id, ndt::type(), nullptr, nullptr},
      {"", any_kind_id, ndt::type(), nullptr, nullptr},
      {"", any_kind_id, ndt::type(), nullptr, nullptr},
      {"", any_kind_id, ndt::type(), nullptr, nullptr},
      {"", any_kind_id, ndt::type(), nullptr, nullptr},
      {"", any_kind_id, ndt::type(), nullptr, nullptr}};

  return infos;
}

ndt::type dynd::default_parse_type_args(type_id_t id, const char *&begin, const char *end,
                                        std::map<std::string, ndt::type> &symtable) {
  ndt::type result, element_type;
  const char *saved_begin = begin;
  nd::buffer args = datashape::parse_type_constr_args(begin, end, symtable);
  if (!args.is_null()) {
    // If there's a '*', there needs to be a following element type.
    if (datashape::parse_token(begin, end, '*')) {
      const char *eltype_saved_begin = begin;
      element_type = datashape::parse(begin, end, symtable);
      if (element_type.is_null()) {
        skip_whitespace_and_pound_comments(begin, end);
        throw datashape::internal_parse_error(eltype_saved_begin, "expected a dynd type");
      }
    }
    const vector<id_info> &infos = detail::infos();
    try {
      result = infos[id].construct_type(id, args, element_type);
    } catch (const dynd::dynd_exception &e) {
      throw datashape::internal_parse_error(saved_begin, e.what());
    } catch (const std::exception &e) {
      throw datashape::internal_parse_error(saved_begin, e.what());
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

DYNDT_API std::pair<type_id_t, const id_info *> dynd::lookup_id_by_name(const std::string &name) {
  // TODO: Create a hash map {name: id_info} for this
  const vector<id_info> &infos = detail::infos();
  for (size_t i = 0, iend = infos.size(); i != iend; ++i) {
    if (infos[i].name == name) {
      return {type_id_t(i), &infos[i]};
    }
  }
  return {uninitialized_id, nullptr};
}

DYNDT_API void dynd::register_known_type_id_constructor(type_id_t id, ndt::type &&singleton_type,
                                                        type_constructor_fn_t construct_type,
                                                        low_level_type_args_parse_fn_t parse_type_args) {
  vector<id_info> &infos = detail::infos();
  if (id < 0 || id >= infos.size()) {
    throw out_of_range("Type ID " + to_string(id) + " is out of range, cannot register its known type ID constructor");
  }
  id_info &ii = infos[id];
  if (ii.singleton_type != ndt::type() || ii.construct_type != nullptr || ii.parse_type_args != nullptr) {
    throw runtime_error("Type ID " + to_string(id) + ", " + ii.name + ", already has type construction registerd");
  }
  ii.singleton_type = std::move(singleton_type);
  if (construct_type != nullptr) {
    ii.construct_type = construct_type;
    // If the type args parser isn't provided, we can put in a default
    ii.parse_type_args = parse_type_args ? parse_type_args : &default_parse_type_args;
  } else if (parse_type_args != nullptr) {
    // If the type args parser is provided, the type constructor needs to be provided too
    throw runtime_error(
        "Type ID " + to_string(id) + ", " + ii.name +
        ", type construction registration needs a type constructor function along with the type args parser");
  }
}
