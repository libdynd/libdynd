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
  m_infos.emplace_back("uninitialized", 0, nullptr,
                       type(reinterpret_cast<const base_type *>(uninitialized_type_id), false), nullptr);

  static const type_id_t bool_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("bool", 1, bool_bases, type(reinterpret_cast<const base_type *>(bool_type_id), false), nullptr);

  static const type_id_t int_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("int8", 1, int_bases, type(reinterpret_cast<const base_type *>(int8_type_id), false), nullptr);
  m_infos.emplace_back("int16", 1, int_bases, type(reinterpret_cast<const base_type *>(int16_type_id), false), nullptr);
  m_infos.emplace_back("int32", 1, int_bases, type(reinterpret_cast<const base_type *>(int32_type_id), false), nullptr);
  m_infos.emplace_back("int64", 1, int_bases, type(reinterpret_cast<const base_type *>(int64_type_id), false), nullptr);
  m_infos.emplace_back("int128", 1, int_bases, type(reinterpret_cast<const base_type *>(int128_type_id), false),
                       nullptr);

  static const type_id_t uint_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("uint8", 1, uint_bases, type(reinterpret_cast<const base_type *>(uint8_type_id), false),
                       nullptr);
  m_infos.emplace_back("uint16", 1, uint_bases, type(reinterpret_cast<const base_type *>(uint16_type_id), false),
                       nullptr);
  m_infos.emplace_back("uint32", 1, uint_bases, type(reinterpret_cast<const base_type *>(uint32_type_id), false),
                       nullptr);
  m_infos.emplace_back("uint64", 1, uint_bases, type(reinterpret_cast<const base_type *>(uint64_type_id), false),
                       nullptr);
  m_infos.emplace_back("uint128", 1, uint_bases, type(reinterpret_cast<const base_type *>(uint128_type_id), false),
                       nullptr);

  static const type_id_t float_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("float16", 1, float_bases, type(reinterpret_cast<const base_type *>(float16_type_id), false),
                       nullptr);
  m_infos.emplace_back("float32", 1, float_bases, type(reinterpret_cast<const base_type *>(float32_type_id), false),
                       nullptr);
  m_infos.emplace_back("float64", 1, float_bases, type(reinterpret_cast<const base_type *>(float64_type_id), false),
                       nullptr);
  m_infos.emplace_back("float128", 1, float_bases, type(reinterpret_cast<const base_type *>(float128_type_id), false),
                       nullptr);

  static const type_id_t complex_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("complex32", 1, complex_bases,
                       type(reinterpret_cast<const base_type *>(complex_float32_type_id), false), nullptr);
  m_infos.emplace_back("complex64", 1, complex_bases,
                       type(reinterpret_cast<const base_type *>(complex_float64_type_id), false), nullptr);

  static const type_id_t void_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("void", 1, void_bases, type(reinterpret_cast<const base_type *>(void_type_id), false), nullptr);

  // ...
  m_infos.emplace_back("Any", 0, nullptr, make_type<any_kind_type>(), nullptr);

  //
  insert("Scalar", any_kind_type_id, make_type<scalar_kind_type>(), nullptr);
  insert("Dim", any_kind_type_id, type(), nullptr);

  insert("Bool", scalar_kind_type_id, type(), nullptr);
  insert("Int", scalar_kind_type_id, type(), nullptr);
  insert("UInt", scalar_kind_type_id, type(), nullptr);
  insert("Float", scalar_kind_type_id, type(), nullptr);
  insert("Complex", scalar_kind_type_id, type(), nullptr);

  // ...
  insert("pointer", any_kind_type_id, pointer_type::make(any_kind_type::make()), nullptr);
  insert("array", any_kind_type_id, type(), nullptr);
  insert("bytes", any_kind_type_id, bytes_type::make(), nullptr);
  insert("fixed_bytes", any_kind_type_id, fixed_bytes_kind_type::make(), nullptr);
  insert("char", any_kind_type_id, make_type<char_type>(), nullptr);
  insert("string", any_kind_type_id, make_type<string_type>(), nullptr);
  insert("fixed_string", any_kind_type_id, fixed_string_kind_type::make(), nullptr);
  insert("Categorical", any_kind_type_id, categorical_kind_type::make(), nullptr);
  insert("date", any_kind_type_id, date_type::make(), nullptr);
  insert("time", any_kind_type_id, time_type::make(), nullptr);
  insert("datetime", any_kind_type_id, datetime_type::make(), nullptr);
  insert("busdate", any_kind_type_id, type(), nullptr);
  insert("Fixed", any_kind_type_id, fixed_dim_kind_type::make(any_kind_type::make()), nullptr);
  insert("var", any_kind_type_id, var_dim_type::make(any_kind_type::make()), nullptr);
  insert("tuple", scalar_kind_type_id, tuple_type::make(true), nullptr);
  insert("struct", tuple_type_id, struct_type::make(true), nullptr);
  insert("option", any_kind_type_id, type(), nullptr);
  insert("C", any_kind_type_id, type(), nullptr);
  insert("adapt", any_kind_type_id, type(), nullptr);
  insert("convert", any_kind_type_id, type(), nullptr);
  insert("view", any_kind_type_id, type(), nullptr);
  insert("cuda_host", any_kind_type_id, type(), nullptr);
  insert("cuda_device", any_kind_type_id, type(), nullptr);
  insert("expr", any_kind_type_id, type(), nullptr);
  insert("type", any_kind_type_id, make_type<type_type>(), nullptr);
}

ndt::type_registry::~type_registry()
{
  for (auto iter = m_infos.begin() + any_kind_type_id; iter != m_infos.end(); ++iter) {
    delete[] iter->bases;
    iter->bases = nullptr;
  }
}

size_t ndt::type_registry::size() const { return m_infos.size(); }

type_id_t ndt::type_registry::insert(const char *name, type_id_t base_tp_id, const type &kind_tp, type_make_t construct)
{
  type_id_t tp_id = static_cast<type_id_t>(size());
  const type_info &base_tp_info = m_infos[base_tp_id];

  size_t nbases = base_tp_info.nbases + 1;
  type_id_t *bases = new type_id_t[nbases]{base_tp_id};
  memcpy(bases + 1, base_tp_info.bases, base_tp_info.nbases);

  m_infos.emplace_back(name, nbases, bases, kind_tp, construct);

  return tp_id;
}

const ndt::type_info &ndt::type_registry::operator[](type_id_t tp_id) const { return m_infos[tp_id]; }

class ndt::type_registry ndt::type_registry;
