//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <bitset>

#include <dynd/type.hpp>

namespace dynd {

struct id_info {
  std::string name;
  type_id_t base_id;

  id_info() = default;

  id_info(const char *name, type_id_t base_id) : name(name), base_id(base_id) {}
};

namespace detail {

  extern DYNDT_API std::vector<id_info> &infos();

} // namespace dynd::detail

DYNDT_API type_id_t new_id(const char *name, type_id_t base_id);

} // namespace dynd
