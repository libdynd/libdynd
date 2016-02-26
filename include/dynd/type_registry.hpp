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
    const type_id_t *_bases;
    type kind_tp;

    std::vector<type_id_t> m_bases;

    std::bitset<64> bits;

    type_info(size_t nbases, const type_id_t *bases, const type &kind_tp)
        : nbases(nbases), _bases(bases), kind_tp(kind_tp), bits(0)
    {
      for (size_t i = 0; i < nbases; ++i) {
        type_id_t base_id = _bases[i];
        bits |= (1L << base_id);
        this->m_bases.push_back(_bases[i]);
      }
    }

    const std::vector<type_id_t> &bases() const { return m_bases; }
  };

  extern DYND_API class type_registry {
    std::vector<type_info> m_infos;

  public:
    type_registry();

    ~type_registry();

    DYND_API size_t size() const;

    type_id_t min() const { return uninitialized_id; }

    type_id_t max() const { return static_cast<type_id_t>(size() - 1); }

    DYND_API type_id_t insert(type_id_t base_id, const type &kind_tp);

    DYND_API const type_info &operator[](type_id_t tp_id) const;
  } type_registry;

} // namespace dynd::ndt
} // namespace dynd
