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
  m_infos.emplace_back("uninitialized", 0, nullptr, nullptr,
                       type(reinterpret_cast<const base_type *>(uninitialized_type_id), false));

  static const type_id_t bool_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("bool", 1, bool_bases, nullptr, type(reinterpret_cast<const base_type *>(bool_type_id), false));

  static const type_id_t int_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("int8", 1, int_bases, nullptr, type(reinterpret_cast<const base_type *>(int8_type_id), false));
  m_infos.emplace_back("int16", 1, int_bases, nullptr, type(reinterpret_cast<const base_type *>(int16_type_id), false));
  m_infos.emplace_back("int32", 1, int_bases, nullptr, type(reinterpret_cast<const base_type *>(int32_type_id), false));
  m_infos.emplace_back("int64", 1, int_bases, nullptr, type(reinterpret_cast<const base_type *>(int64_type_id), false));
  m_infos.emplace_back("int128", 1, int_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(int128_type_id), false));

  static const type_id_t uint_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("uint8", 1, uint_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(uint8_type_id), false));
  m_infos.emplace_back("uint16", 1, uint_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(uint16_type_id), false));
  m_infos.emplace_back("uint32", 1, uint_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(uint32_type_id), false));
  m_infos.emplace_back("uint64", 1, uint_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(uint64_type_id), false));
  m_infos.emplace_back("uint128", 1, uint_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(uint128_type_id), false));

  static const type_id_t float_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("float16", 1, float_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(float16_type_id), false));
  m_infos.emplace_back("float32", 1, float_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(float32_type_id), false));
  m_infos.emplace_back("float64", 1, float_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(float64_type_id), false));
  m_infos.emplace_back("float128", 1, float_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(float128_type_id), false));

  static const type_id_t complex_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("complex32", 1, complex_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(complex_float32_type_id), false));
  m_infos.emplace_back("complex64", 1, complex_bases, nullptr,
                       type(reinterpret_cast<const base_type *>(complex_float64_type_id), false));

  static const type_id_t void_bases[1] = {any_kind_type_id};
  m_infos.emplace_back("void", 1, void_bases, nullptr, type(reinterpret_cast<const base_type *>(void_type_id), false));

  // ...
  m_infos.emplace_back("Any", 0, nullptr, nullptr, make_type<any_kind_type>());

  //
  insert("Scalar", any_kind_type_id, nullptr, make_type<scalar_kind_type>());

  // ...
  insert("pointer", any_kind_type_id, nullptr, pointer_type::make(any_kind_type::make()));
  insert("array", any_kind_type_id, nullptr, type());
  insert("bytes", any_kind_type_id, nullptr, bytes_type::make());
  insert("fixed_bytes", any_kind_type_id, nullptr, fixed_bytes_kind_type::make());
  insert("char", any_kind_type_id, nullptr, make_type<char_type>());
  insert("string", any_kind_type_id, nullptr, make_type<string_type>());
  insert("fixed_string", any_kind_type_id, nullptr, fixed_string_kind_type::make());
  insert("Categorical", any_kind_type_id, nullptr, categorical_kind_type::make());
  insert("date", any_kind_type_id, nullptr, date_type::make());
  insert("time", any_kind_type_id, nullptr, time_type::make());
  insert("datetime", any_kind_type_id, nullptr, datetime_type::make());
  insert("busdate", any_kind_type_id, nullptr, type());
  insert("Fixed", any_kind_type_id, nullptr, fixed_dim_kind_type::make(any_kind_type::make()));
  insert("var", any_kind_type_id, nullptr, var_dim_type::make(any_kind_type::make()));
  insert("tuple", any_kind_type_id, nullptr, tuple_type::make(true));
  insert("struct", tuple_type_id, nullptr, struct_type::make(true));
  insert("option", any_kind_type_id, nullptr, type());
  insert("C", any_kind_type_id, nullptr, type());
  insert("adapt", any_kind_type_id, nullptr, type());
  insert("convert", any_kind_type_id, nullptr, type());
  insert("view", any_kind_type_id, nullptr, type());
  insert("cuda_host", any_kind_type_id, nullptr, type());
  insert("cuda_device", any_kind_type_id, nullptr, type());
  insert("expr", any_kind_type_id, nullptr, type());
  insert("type", any_kind_type_id, nullptr, make_type<type_type>());
}

ndt::type_registry::~type_registry()
{
  for (auto iter = m_infos.begin() + any_kind_type_id; iter != m_infos.end(); ++iter) {
    delete[] iter->bases;
    iter->bases = nullptr;
  }
}

size_t ndt::type_registry::size() const { return m_infos.size(); }

type_id_t ndt::type_registry::insert(const char *name, type_id_t base_id, type_make_t construct, const type &kind)
{
  size_t nbases = m_infos[base_id].nbases + 1;
  type_id_t *bases = new type_id_t[nbases];
  bases[0] = base_id;
  memcpy(bases + 1, m_infos[base_id].bases, m_infos[base_id].nbases);

  m_infos.emplace_back(name, nbases, bases, construct, kind);
  return static_cast<type_id_t>(size() - 1);
}

const ndt::type_info &ndt::type_registry::operator[](type_id_t tp_id) const { return m_infos[tp_id]; }

class ndt::type_registry ndt::type_registry;
