//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    arrfunc outer(const arrfunc &child);

    ndt::type outer_make_type(const arrfunc_type *child_tp);

    intptr_t outer_instantiate(
        const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
        void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
        const char *dst_arrmeta, const ndt::type *src_tp,
        const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
        const eval::eval_context *ectx, const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars);

    int outer_resolve_dst_type(
        const arrfunc_type_data *self, const arrfunc_type *self_tp,
        intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
        ndt::type &dst_tp, const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars);


    int outer_resolve_dst_type_with_child(
        const arrfunc_type_data *child, const arrfunc_type *child_tp,
        intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
        ndt::type &dst_tp, const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd