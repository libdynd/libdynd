//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type_registry.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

ndt::type_registry::type_registry()
{
  m_infos.emplace_back();

  m_infos.emplace_back(make_type<any_kind_type>());
  insert(base_id_of<scalar_kind_id>::value, make_type<scalar_kind_type>());

  insert(base_id_of<bool_kind_id>::value, type());
  insert(base_id_of<bool_id>::value, type(reinterpret_cast<const base_type *>(bool_id), false));

  insert(base_id_of<int_kind_id>::value, type());
  insert(base_id_of<int8_id>::value, type(reinterpret_cast<const base_type *>(int8_id), false));
  insert(base_id_of<int16_id>::value, type(reinterpret_cast<const base_type *>(int16_id), false));
  insert(base_id_of<int32_id>::value, type(reinterpret_cast<const base_type *>(int32_id), false));
  insert(base_id_of<int64_id>::value, type(reinterpret_cast<const base_type *>(int64_id), false));
  insert(base_id_of<int128_id>::value, type(reinterpret_cast<const base_type *>(int128_id), false));

  insert(base_id_of<uint_kind_id>::value, type());
  insert(base_id_of<uint8_id>::value, type(reinterpret_cast<const base_type *>(uint8_id), false));
  insert(base_id_of<uint16_id>::value, type(reinterpret_cast<const base_type *>(uint16_id), false));
  insert(base_id_of<uint32_id>::value, type(reinterpret_cast<const base_type *>(uint32_id), false));
  insert(base_id_of<uint64_id>::value, type(reinterpret_cast<const base_type *>(uint64_id), false));
  insert(base_id_of<uint128_id>::value, type(reinterpret_cast<const base_type *>(uint128_id), false));

  insert(base_id_of<float_kind_id>::value, type());
  insert(base_id_of<float16_id>::value, type(reinterpret_cast<const base_type *>(float16_id), false));
  insert(base_id_of<float32_id>::value, type(reinterpret_cast<const base_type *>(float32_id), false));
  insert(base_id_of<float64_id>::value, type(reinterpret_cast<const base_type *>(float64_id), false));
  insert(base_id_of<float128_id>::value, type(reinterpret_cast<const base_type *>(float128_id), false));

  insert(base_id_of<complex_kind_id>::value, type());
  insert(base_id_of<complex_float32_id>::value, type(reinterpret_cast<const base_type *>(complex_float32_id), false));
  insert(base_id_of<complex_float64_id>::value, type(reinterpret_cast<const base_type *>(complex_float64_id), false));

  insert(base_id_of<void_id>::value, type(reinterpret_cast<const base_type *>(void_id), false));

  insert(base_id_of<dim_kind_id>::value, type());

  insert(base_id_of<bytes_kind_id>::value, type());
  insert(base_id_of<fixed_bytes_id>::value, fixed_bytes_kind_type::make());
  insert(base_id_of<bytes_id>::value, bytes_type::make());

  insert(base_id_of<string_kind_id>::value, type());
  insert(base_id_of<fixed_string_id>::value, fixed_string_kind_type::make());
  insert(base_id_of<char_id>::value, make_type<char_type>());
  insert(base_id_of<string_id>::value, make_type<string_type>());

  insert(base_id_of<tuple_id>::value, tuple_type::make(true));
  insert(base_id_of<struct_id>::value, struct_type::make(true));

  insert(base_id_of<fixed_dim_id>::value, base_fixed_dim_type::make(any_kind_type::make()));
  insert(base_id_of<var_dim_id>::value, var_dim_type::make(any_kind_type::make()));

  insert(scalar_kind_id, categorical_kind_type::make());          // categorical_id
  insert(any_kind_id, type("?Any"));                              // option_id
  insert(any_kind_id, pointer_type::make(any_kind_type::make())); // pointer_id
  insert(any_kind_id, type());                                    // memory_id

  insert(base_id_of<type_id>::value, make_type<type_type>());
  insert(base_id_of<array_id>::value, type());
  insert(base_id_of<callable_id>::value, type());

  insert(any_kind_id, type());  // expr_kind_id
  insert(expr_kind_id, type()); // adapt_id
  insert(expr_kind_id, type()); // expr_id

  insert(any_kind_id, type()); // cuda_host_id
  insert(any_kind_id, type()); // cuda_device_id

  insert(any_kind_id, type()); // kind_sym_id
  insert(any_kind_id, type()); // int_sym_id

  insert(any_kind_id, type()); // typevar_id
  insert(any_kind_id, type()); // typevar_dim_id
  insert(any_kind_id, type()); // typevar_constructed_id
  insert(any_kind_id, type()); // pow_dimsym_id
  insert(any_kind_id, type()); // ellipsis_dim_id
  insert(any_kind_id, type()); // dim_fragment_id
}

DYND_API size_t ndt::type_registry::size() const { return m_infos.size(); }

DYND_API type_id_t ndt::type_registry::insert(type_id_t base_id, const type &tp)
{
  type_id_t id = static_cast<type_id_t>(size());

  vector<type_id_t> base_ids{base_id};
  for (type_id_t id : m_infos[base_id].get_base_ids()) {
    base_ids.push_back(id);
  }

  m_infos.emplace_back(tp, base_ids);
  m_infos[id].bits |= 1L << id;

  return id;
}

DYND_API const id_info &ndt::type_registry::operator[](type_id_t id) const
{
  if (id >= static_cast<type_id_t>(size())) {
    throw runtime_error("invalid type id");
  }

  return m_infos[id];
}

DYND_API class ndt::type_registry ndt::type_registry;
