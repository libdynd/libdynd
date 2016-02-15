//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

namespace dynd {
namespace ndt {

  struct type_info {
    size_t nbases;
    const type_id_t *bases;
    type kind_tp;

    type_info(size_t nbases, const type_id_t *bases, const type &kind_tp)
        : nbases(nbases), bases(bases), kind_tp(kind_tp)
    {
    }
  };

  extern DYND_API class type_registry {
    std::vector<type_info> m_infos;

  public:
    type_registry();

    ~type_registry();

    DYND_API size_t size() const;

    type_id_t max() const { return static_cast<type_id_t>(size() - 1); }

    DYND_API type_id_t insert(type_id_t base_id, const type &kind_tp);

    DYND_API const type_info &operator[](type_id_t tp_id) const;
  } type_registry;

} // namespace dynd::ndt
} // namespace dynd
