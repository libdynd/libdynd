//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <numeric>

#include <dynd/flat_iterator.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/multidispatch_kernel.hpp>

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

    arrfunc multidispatch(const ndt::type &self_tp, intptr_t size,
                          const arrfunc *children, const arrfunc &default_child,
                          bool own_children, intptr_t i0 = 0);

    arrfunc multidispatch_by_type_id(const ndt::type &self_tp,
                                     const std::vector<arrfunc> &children);

    inline arrfunc multidispatch_by_type_id(const ndt::type &self_tp,
                                            intptr_t size,
                                            const arrfunc *children,
                                            const arrfunc &default_child,
                                            bool own_children, intptr_t i0 = 0)
    {
      return multidispatch(self_tp, size, children, default_child, own_children,
                           i0);
    }

    template <int N0>
    arrfunc multidispatch(const ndt::type &self_tp,
                          const arrfunc (&children)[N0],
                          const arrfunc &default_child, intptr_t i0 = 0)
    {
      return multidispatch(self_tp, N0, children, default_child, false, i0);
    }

    template <typename ContainerType, int N = ndim<ContainerType>::value>
    arrfunc multidispatch(const ndt::type &self_tp,
                          const ContainerType &children,
                          const arrfunc &DYND_UNUSED(default_child),
                          const std::vector<intptr_t> &permutation)
    {
      for (auto it = dynd::begin<N>(children); it != it.end(); ++it) {
        const arrfunc &child = *it;
        if (!child.is_null()) {
          std::map<string, ndt::type> tp_vars;
          if (!self_tp.match(child.get_array_type(), tp_vars)) {
            throw std::invalid_argument("could not match arrfuncs");
          }
        }
      }

      struct static_data {
        const ContainerType &children;
        intptr_t permutation[N];

        static_data(const ContainerType &children, const intptr_t *permutation)
            : children(children)
        {
          std::memcpy(this->permutation, permutation,
                      sizeof(this->permutation));
        }

        arrfunc operator()(const ndt::type &dst_tp, intptr_t nsrc,
                           const ndt::type *src_tp) const
        {
          std::vector<ndt::type> tp;
          tp.push_back(dst_tp);
          for (int j = 0; j < nsrc; ++j) {
            tp.push_back(src_tp[j]);
          }
          ndt::type *new_src_tp = tp.data() + 1;

          intptr_t index[N];
          for (intptr_t j = 0; j < N; ++j) {
            index[j] = new_src_tp[permutation[j]].get_type_id();
          }

          return at(children, index);
        }
      };

      return arrfunc::make<multidispatch_kernel<static_data>>(
          self_tp, static_data(children, permutation.data()), 0);
    }

    template <typename ContainerType, int N = ndim<ContainerType>::value>
    arrfunc multidispatch(const ndt::type &self_tp,
                          const ContainerType &children,
                          const arrfunc &default_child)
    {
      std::vector<intptr_t> permutation(N);
      std::iota(permutation.begin(), permutation.end(), 0);

      return multidispatch(self_tp, children, default_child, permutation);
    }

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
