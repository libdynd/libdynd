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
                          const std::vector<arrfunc> &children);

    intptr_t multidispatch_instantiate(
        const arrfunc_type_data *self, const arrfunc_type *af_tp, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx,
        const array &kwds, const std::map<string, ndt::type> &tp_vars);

    void multidispatch_resolve_option_values(
        const arrfunc_type_data *self, const arrfunc_type *af_tp, intptr_t nsrc,
        const ndt::type *src_tp, nd::array &kwds,
        const std::map<nd::string, ndt::type> &tp_vars);

    int multidispatch_resolve_dst_type(
        const arrfunc_type_data *self, const arrfunc_type *af_tp, intptr_t nsrc,
        const ndt::type *src_tp, int throw_on_error, ndt::type &out_dst_tp,
        const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
