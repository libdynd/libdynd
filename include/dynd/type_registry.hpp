//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <bitset>

#include <dynd/type.hpp>

namespace dynd {

namespace ndt {

  extern DYNDT_API class type_registry {
  public:
    type_registry();
  } type_registry;

} // namespace dynd::ndt

struct id_info {
  std::vector<type_id_t> base_ids;
  std::vector<char> is_base_id;

  id_info() = default;

  id_info(type_id_t id, size_t size = 128) : is_base_id(size) { is_base_id[id] = true; }

  id_info(type_id_t id, const std::vector<type_id_t> &base_ids, size_t size = 128)
      : base_ids(base_ids), is_base_id(size)
  {
    is_base_id[id] = true;
    for (type_id_t base_id : this->base_ids) {
      is_base_id[base_id] = true;
    }
  }
};

namespace detail {

  DYNDT_API std::vector<id_info> &infos();

} // namespace dynd::detail

DYNDT_API type_id_t new_id(type_id_t base_id);

inline type_id_t min_id() { return static_cast<type_id_t>(1); }

inline type_id_t max_id()
{
  const std::vector<id_info> &infos = detail::infos();
  return static_cast<type_id_t>(infos.size() - 1);
}

inline type_id_t base_id(type_id_t id)
{
  const std::vector<id_info> &infos = detail::infos();
  return infos[id].base_ids.front();
}

template <type_id_t ID>
std::enable_if_t<ID == any_kind_id, std::array<type_id_t, 0>> base_ids()
{
  return {};
}

template <type_id_t ID>
std::enable_if_t<ID != any_kind_id, std::vector<type_id_t>> base_ids()
{
  std::vector<type_id_t> res{base_id_of<ID>::value};
  for (type_id_t base_id : base_ids<base_id_of<ID>::value>()) {
    res.push_back(base_id);
  }

  return res;
}

inline const std::vector<type_id_t> &base_ids(type_id_t id)
{
  const std::vector<id_info> &infos = detail::infos();
  return infos[id].base_ids;
}

inline bool is_base_id_of(type_id_t base_id, type_id_t id)
{
  const std::vector<id_info> &infos = detail::infos();
  return infos[id].is_base_id[base_id] != 0;
}

} // namespace dynd
