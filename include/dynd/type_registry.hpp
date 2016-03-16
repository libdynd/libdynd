//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <bitset>

#include <dynd/type.hpp>

namespace dynd {

class id_info {
public:
  ndt::type m_tp;
  std::vector<type_id_t> m_base_ids;
  std::vector<char> m_is_base_id;

  id_info() = default;

  id_info(type_id_t id, const ndt::type &tp, size_t size = 128) : m_tp(tp), m_is_base_id(size)
  {
    m_is_base_id[id] = true;
  }

  id_info(type_id_t id, const ndt::type &tp, const std::vector<type_id_t> &base_ids, size_t size = 128)
      : m_tp(tp), m_base_ids(base_ids), m_is_base_id(size)
  {
    m_is_base_id[id] = true; 
    for (type_id_t base_id : m_base_ids) {
      m_is_base_id[base_id] = true;
    }
  }

  type_id_t get_base_id() const { return m_base_ids.front(); }

  const std::vector<type_id_t> &get_base_ids() const { return m_base_ids; }

  const ndt::type &get_type() const { return m_tp; }

  bool is_base_id(type_id_t id) const { return m_is_base_id[id] != 0; }
};

namespace ndt {

  extern DYNDT_API class type_registry {
    std::vector<id_info> m_infos;

  public:
    type_registry();

    DYNDT_API size_t size() const;

    type_id_t min() const { return static_cast<type_id_t>(1); }

    type_id_t max() const { return static_cast<type_id_t>(size() - 1); }

    DYNDT_API type_id_t insert(type_id_t base_id, const type &kind_tp);

    DYNDT_API const id_info &operator[](type_id_t id) const { return m_infos[id]; }

  } type_registry;

} // namespace dynd::ndt

inline bool is_base_id_of(type_id_t base_id, type_id_t id) { return ndt::type_registry[id].is_base_id(base_id); }

} // namespace dynd
