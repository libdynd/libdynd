//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type_registry.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_dim_kind_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

ndt::type_registry::type_registry()
{
  m_infos.emplace_back(
      0, nullptr, type(reinterpret_cast<const base_type *>(uninitialized_type_id), false)); // uninitialized_type_id

  static const type_id_t bool_bases[3] = {bool_kind_type_id, scalar_kind_type_id, any_kind_type_id};
  m_infos.emplace_back(3, bool_bases, type(reinterpret_cast<const base_type *>(bool_type_id), false)); // bool_type_id

  static const type_id_t int_bases[3] = {int_kind_type_id, scalar_kind_type_id, any_kind_type_id};
  m_infos.emplace_back(3, int_bases, type(reinterpret_cast<const base_type *>(int8_type_id), false));  // int8_type_id
  m_infos.emplace_back(3, int_bases, type(reinterpret_cast<const base_type *>(int16_type_id), false)); // int16_type_id
  m_infos.emplace_back(3, int_bases, type(reinterpret_cast<const base_type *>(int32_type_id), false)); // int32_type_id
  m_infos.emplace_back(3, int_bases, type(reinterpret_cast<const base_type *>(int64_type_id), false)); // int64_type_id
  m_infos.emplace_back(3, int_bases,
                       type(reinterpret_cast<const base_type *>(int128_type_id), false)); // int128_type_id

  static const type_id_t uint_bases[3] = {uint_kind_type_id, scalar_kind_type_id, any_kind_type_id};
  m_infos.emplace_back(3, uint_bases, type(reinterpret_cast<const base_type *>(uint8_type_id), false)); // uint8_type_id
  m_infos.emplace_back(3, uint_bases,
                       type(reinterpret_cast<const base_type *>(uint16_type_id), false)); // uint16_type_id
  m_infos.emplace_back(3, uint_bases,
                       type(reinterpret_cast<const base_type *>(uint32_type_id), false)); // uint32_type_id
  m_infos.emplace_back(3, uint_bases,
                       type(reinterpret_cast<const base_type *>(uint64_type_id), false)); // uint64_type_id
  m_infos.emplace_back(3, uint_bases,
                       type(reinterpret_cast<const base_type *>(uint128_type_id), false)); // uint128_type_id

  static const type_id_t float_bases[3] = {float_kind_type_id, scalar_kind_type_id, any_kind_type_id};
  m_infos.emplace_back(3, float_bases,
                       type(reinterpret_cast<const base_type *>(float16_type_id), false)); // float16_type_id
  m_infos.emplace_back(3, float_bases,
                       type(reinterpret_cast<const base_type *>(float32_type_id), false)); // float32_type_id
  m_infos.emplace_back(3, float_bases,
                       type(reinterpret_cast<const base_type *>(float64_type_id), false)); // float64_type_id
  m_infos.emplace_back(3, float_bases,
                       type(reinterpret_cast<const base_type *>(float128_type_id), false)); // float128_type_id

  static const type_id_t complex_bases[3] = {complex_kind_type_id, scalar_kind_type_id, any_kind_type_id};
  m_infos.emplace_back(3, complex_bases,
                       type(reinterpret_cast<const base_type *>(complex_float32_type_id), false)); // complex32_type_id
  m_infos.emplace_back(3, complex_bases,
                       type(reinterpret_cast<const base_type *>(complex_float64_type_id), false)); // complex64_type_id

  static const type_id_t void_bases[1] = {any_kind_type_id};
  m_infos.emplace_back(1, void_bases, type(reinterpret_cast<const base_type *>(void_type_id), false)); // void_type_id

  m_infos.emplace_back(0, nullptr, make_type<any_kind_type>()); // any_kind_type_id
  insert(any_kind_type_id, make_type<scalar_kind_type>());      // scalar_kind_type_id
  insert(any_kind_type_id, type());                             // dim_kind_type_id

  insert(scalar_kind_type_id, type()); // bool_kind_type_id
  insert(scalar_kind_type_id, type()); // int_kind_type_id
  insert(scalar_kind_type_id, type()); // uint_kind_type_id
  insert(scalar_kind_type_id, type()); // float_kind_type_id
  insert(scalar_kind_type_id, type()); // complex_kind_type_id

  insert(scalar_kind_type_id, fixed_bytes_kind_type::make()); // fixed_bytes_type_id
  insert(scalar_kind_type_id, bytes_type::make());            // bytes_type_id

  insert(scalar_kind_type_id, fixed_string_kind_type::make()); // fixed_string_type_id
  insert(scalar_kind_type_id, make_type<char_type>());         // char_type_id
  insert(scalar_kind_type_id, make_type<string_type>());       // string_type_id

  insert(scalar_kind_type_id, date_type::make());     // date_type_id
  insert(scalar_kind_type_id, time_type::make());     // time_type_id
  insert(scalar_kind_type_id, datetime_type::make()); // datetime_type_id
  insert(scalar_kind_type_id, type());                // busdate_type_id

  insert(scalar_kind_type_id, tuple_type::make(true)); // tuple_type_id
  insert(tuple_type_id, struct_type::make(true));      // struct_type_id

  insert(dim_kind_type_id, fixed_dim_kind_type::make(any_kind_type::make())); // fixed_dim_type_id
  insert(dim_kind_type_id, var_dim_type::make(any_kind_type::make()));        // var_dim_type_id

  insert(any_kind_type_id, categorical_kind_type::make());             // categorical_type_id
  insert(any_kind_type_id, type("?Any"));                              // option_type_id
  insert(any_kind_type_id, pointer_type::make(any_kind_type::make())); // pointer_type_id

  insert(any_kind_type_id, make_type<type_type>()); // type_type_id
  insert(any_kind_type_id, type());                 // array_type_id
  insert(any_kind_type_id, type());                 // callable_type_id

  insert(any_kind_type_id, type()); // adapt_type_id
  insert(any_kind_type_id, type()); // expr_type_id
  insert(any_kind_type_id, type()); // convert_type_id
  insert(any_kind_type_id, type()); // view_type_id

  insert(any_kind_type_id, type()); // c_contiguous_type_id

  insert(any_kind_type_id, type()); // cuda_host_type_id
  insert(any_kind_type_id, type()); // cuda_device_type_id

  insert(any_kind_type_id, type()); // kind_sym_type_id
  insert(any_kind_type_id, type()); // int_sym_type_id

  insert(any_kind_type_id, type()); // typevar_type_id
  insert(any_kind_type_id, type()); // typevar_dim_type_id
  insert(any_kind_type_id, type()); // typevar_constructed_type_id
  insert(any_kind_type_id, type()); // pow_dimsym_type_id
  insert(any_kind_type_id, type()); // ellipsis_dim_type_id
  insert(any_kind_type_id, type()); // dim_fragment_type_id
}

ndt::type_registry::~type_registry()
{
  for (auto iter = m_infos.begin() + any_kind_type_id; iter != m_infos.end(); ++iter) {
    delete[] iter->bases;
    iter->bases = nullptr;
  }
}

DYND_API size_t ndt::type_registry::size() const { return m_infos.size(); }

DYND_API type_id_t ndt::type_registry::insert(type_id_t base_tp_id, const type &kind_tp)
{
  type_id_t tp_id = static_cast<type_id_t>(size());
  const type_info &base_tp_info = m_infos[base_tp_id];

  size_t nbases = base_tp_info.nbases + 1;
  type_id_t *bases = new type_id_t[nbases]{base_tp_id};
  memcpy(bases + 1, base_tp_info.bases, base_tp_info.nbases);

  m_infos.emplace_back(nbases, bases, kind_tp);

  return tp_id;
}

DYND_API const ndt::type_info &ndt::type_registry::operator[](type_id_t id) const
{
  if (id >= static_cast<type_id_t>(size())) {
    throw runtime_error("invalid type id");
  }

  return m_infos[id];
}

DYND_API class ndt::type_registry ndt::type_registry;
