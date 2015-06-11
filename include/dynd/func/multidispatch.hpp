//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/multidispatch.hpp>

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

    arrfunc multidispatch_by_type_id(const ndt::type &self_tp, intptr_t size,
                                     const arrfunc *children,
                                     const arrfunc &default_child,
                                     bool own_children, intptr_t i0 = 0);

    template <int N0, int N1>
    arrfunc multidispatch_by_type_id(const ndt::type &self_tp,
                                     const arrfunc (&children)[N0][N1],
                                     const arrfunc &DYND_UNUSED(default_child))
    {
      for (int i0 = 0; i0 < N0; ++i0) {
        for (int i1 = 0; i1 < N1; ++i1) {
          const arrfunc &child = children[i0][i1];
          if (!child.is_null()) {
            std::cout << child << std::endl;

            std::map<string, ndt::type> tp_vars;
            if (!self_tp.match(child.get_array_type(), tp_vars)) {
              throw std::invalid_argument("could not match arrfuncs");
            }
          }
        }
      }

      return arrfunc::make<new_multidispatch_by_type_id_kernel<N0, N1>>(
          ndt::type("(Any, Any) -> Any"), &children, 0);
    }

    arrfunc multidispatch_by_type_id(const ndt::type &self_tp,
                                     const std::vector<arrfunc> &children);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
