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

  id_info() = default;

  id_info(const char *name) : name(name) {}
};

namespace detail {

  extern DYNDT_API std::vector<id_info> &infos();

} // namespace dynd::detail

DYNDT_API type_id_t new_id(const char *name);

} // namespace dynd
