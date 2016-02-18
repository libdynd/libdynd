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
#include <dynd/types/fixed_dim_kind_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

ndt::type_registry::type_registry()
{
  m_infos.emplace_back(0, nullptr,
                       type(reinterpret_cast<const base_type *>(uninitialized_id), false)); // uninitialized_id

  static const type_id_t bool_base_ids[3] = {bool_kind_id, scalar_kind_id, any_kind_id};
  m_infos.emplace_back(3, bool_base_ids, type(reinterpret_cast<const base_type *>(bool_id), false)); // bool_id

  static const type_id_t int_bases[3] = {int_kind_id, scalar_kind_id, any_kind_id};
  m_infos.emplace_back(3, int_bases, type(reinterpret_cast<const base_type *>(int8_id), false));   // int8_id
  m_infos.emplace_back(3, int_bases, type(reinterpret_cast<const base_type *>(int16_id), false));  // int16_id
  m_infos.emplace_back(3, int_bases, type(reinterpret_cast<const base_type *>(int32_id), false));  // int32_id
  m_infos.emplace_back(3, int_bases, type(reinterpret_cast<const base_type *>(int64_id), false));  // int64_id
  m_infos.emplace_back(3, int_bases, type(reinterpret_cast<const base_type *>(int128_id), false)); // int128_id

  static const type_id_t uint_bases[3] = {uint_kind_id, scalar_kind_id, any_kind_id};
  m_infos.emplace_back(3, uint_bases, type(reinterpret_cast<const base_type *>(uint8_id), false));   // uint8_id
  m_infos.emplace_back(3, uint_bases, type(reinterpret_cast<const base_type *>(uint16_id), false));  // uint16_id
  m_infos.emplace_back(3, uint_bases, type(reinterpret_cast<const base_type *>(uint32_id), false));  // uint32_id
  m_infos.emplace_back(3, uint_bases, type(reinterpret_cast<const base_type *>(uint64_id), false));  // uint64_id
  m_infos.emplace_back(3, uint_bases, type(reinterpret_cast<const base_type *>(uint128_id), false)); // uint128_id

  static const type_id_t float_bases[3] = {float_kind_id, scalar_kind_id, any_kind_id};
  m_infos.emplace_back(3, float_bases, type(reinterpret_cast<const base_type *>(float16_id), false));  // float16_id
  m_infos.emplace_back(3, float_bases, type(reinterpret_cast<const base_type *>(float32_id), false));  // float32_id
  m_infos.emplace_back(3, float_bases, type(reinterpret_cast<const base_type *>(float64_id), false));  // float64_id
  m_infos.emplace_back(3, float_bases, type(reinterpret_cast<const base_type *>(float128_id), false)); // float128_id

  static const type_id_t complex_bases[3] = {complex_kind_id, scalar_kind_id, any_kind_id};
  m_infos.emplace_back(3, complex_bases,
                       type(reinterpret_cast<const base_type *>(complex_float32_id), false)); // complex32_id
  m_infos.emplace_back(3, complex_bases,
                       type(reinterpret_cast<const base_type *>(complex_float64_id), false)); // complex64_id

  static const type_id_t void_bases[1] = {any_kind_id};
  m_infos.emplace_back(1, void_bases, type(reinterpret_cast<const base_type *>(void_id), false)); // void_id

  m_infos.emplace_back(0, nullptr, make_type<any_kind_type>()); // any_kind_id
  insert(any_kind_id, make_type<scalar_kind_type>());           // scalar_kind_id
  insert(any_kind_id, type());                                  // dim_kind_id

  insert(scalar_kind_id, type()); // bool_kind_id
  insert(scalar_kind_id, type()); // int_kind_id
  insert(scalar_kind_id, type()); // uint_kind_id
  insert(scalar_kind_id, type()); // float_kind_id
  insert(scalar_kind_id, type()); // complex_kind_id

  insert(scalar_kind_id, type());                       // bytes_kind_id
  insert(bytes_kind_id, fixed_bytes_kind_type::make()); // fixed_bytes_id
  insert(bytes_kind_id, bytes_type::make());            // bytes_id

  insert(scalar_kind_id, type());                         // string_kind_id
  insert(string_kind_id, fixed_string_kind_type::make()); // fixed_string_id
  insert(string_kind_id, make_type<char_type>());         // char_id
  insert(string_kind_id, make_type<string_type>());       // string_id

  insert(scalar_kind_id, tuple_type::make(true)); // tuple_id
  insert(tuple_id, struct_type::make(true));      // struct_id

  insert(dim_kind_id, fixed_dim_kind_type::make(any_kind_type::make())); // fixed_dim_id
  insert(dim_kind_id, var_dim_type::make(any_kind_type::make()));        // var_dim_id

  insert(scalar_kind_id, categorical_kind_type::make());          // categorical_id
  insert(any_kind_id, type("?Any"));                              // option_id
  insert(any_kind_id, pointer_type::make(any_kind_type::make())); // pointer_id
  insert(any_kind_id, type());                                    // memory_id

  insert(scalar_kind_id, make_type<type_type>()); // type_id
  insert(scalar_kind_id, type());                 // array_id
  insert(scalar_kind_id, type());                 // callable_id

  insert(any_kind_id, type());  // expr_kind_id
  insert(expr_kind_id, type()); // adapt_id
  insert(expr_kind_id, type()); // expr_id

  insert(any_kind_id, type()); // c_contiguous_id

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

ndt::type_registry::~type_registry()
{
  for (auto iter = m_infos.begin() + any_kind_id; iter != m_infos.end(); ++iter) {
    delete[] iter->bases;
    iter->bases = nullptr;
  }
}

DYND_API size_t ndt::type_registry::size() const { return m_infos.size(); }

DYND_API type_id_t ndt::type_registry::insert(type_id_t base_id, const type &kind_tp)
{
  type_id_t id = static_cast<type_id_t>(size());
  const type_info &base_tp_info = m_infos[base_id];

  size_t nbases = base_tp_info.nbases + 1;
  type_id_t *bases = new type_id_t[nbases]{base_id};
  memcpy(bases + 1, base_tp_info.bases, base_tp_info.nbases);

  m_infos.emplace_back(nbases, bases, kind_tp);

  return id;
}

DYND_API const ndt::type_info &ndt::type_registry::operator[](type_id_t id) const
{
  if (id >= static_cast<type_id_t>(size())) {
    throw runtime_error("invalid type id");
  }

  return m_infos[id];
}

DYND_API class ndt::type_registry ndt::type_registry;
