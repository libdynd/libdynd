//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Creates a multiple dispatch arrfunc out of a set of arrfuncs. The
     * input arrfuncs must have concrete signatures.
     *
     * \param naf  The number of arrfuncs provided.
     * \param af  The array of input arrfuncs, sized ``naf``.
     */
    arrfunc multidispatch(intptr_t naf, const arrfunc *af);

    inline arrfunc multidispatch(const std::initializer_list<arrfunc> &children)
    {
      return multidispatch(children.size(), children.begin());
    }

    arrfunc multidispatch(const ndt::type &self_tp,
                          const std::vector<arrfunc> &children,
                          const std::vector<std::string> &ignore_vars);

    arrfunc multidispatch(const ndt::type &self_tp,
                          const std::vector<arrfunc> &children);

    template <bool own_children = true>
    arrfunc multidispatch_by_type_id(const ndt::type &self_tp, intptr_t size,
                                     const arrfunc *children,
                                     const arrfunc &default_child, intptr_t i0 = 0);

    arrfunc multidispatch_by_type_id(const ndt::type &self_tp,
                                     const std::vector<arrfunc> &children);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
