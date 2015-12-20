//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

namespace dynd {
namespace ndt {

  struct type_info {
    const char *name;
    size_t nbases;
    const type_id_t *bases;
    type_make_t construct;
    type kind;

    type_info(const char *name, size_t nbases, const type_id_t *bases, type_make_t construct, const type &kind)
        : name(name), nbases(nbases), bases(bases), construct(construct), kind(kind)
    {
    }
  };

  extern class type_registry {
    std::vector<type_info> m_infos;

  public:
    type_registry();

    ~type_registry();

    size_t size() const;

    type_id_t insert(const char *name, type_id_t tp_id, type_make_t construct, const type &kind);

    const type_info &operator[](type_id_t tp_id) const;
  } type_registry;

} // namespace dynd::ndt
} // namespace dynd
