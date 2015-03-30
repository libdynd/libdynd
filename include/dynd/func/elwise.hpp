//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/kernels/elwise.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    arrfunc elwise(const arrfunc &child);

    intptr_t elwise_instantiate(
        const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
        char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
        const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
        const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
        const eval::eval_context *ectx, const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars);

    /**
     * Lifts the provided ckernel, broadcasting it as necessary to execute
     * across the additional dimensions in the ``lifted_types`` array.
     *
     * This version is for 'expr' ckernels.
     *
     * \param child  The arrfunc being lifted
     * \param ckb  The ckernel_builder into which to place the ckernel.
     * \param ckb_offset  Where within the ckernel_builder to place the
     *ckernel.
     * \param dst_tp  The destination type to lift to.
     * \param dst_arrmeta  The destination arrmeta to lift to.
     * \param src_tp  The source types to lift to.
     * \param src_arrmeta  The source arrmetas to lift to.
     * \param kernreq  Either kernel_request_single or kernel_request_strided,
     *                 as required by the caller.
     * \param ectx  The evaluation context.
     */
    intptr_t elwise_instantiate_with_child(
        const arrfunc_type_data *child, const arrfunc_type *child_tp, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
        const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars);

    template <int I>
    intptr_t elwise_instantiate_with_child(
        const arrfunc_type_data *child, const arrfunc_type *child_tp, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
        const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars);

    template <type_id_t dst_dim_id, type_id_t src_dim_id, int I>
    intptr_t elwise_instantiate_with_child(
        const arrfunc_type_data *child, const arrfunc_type *child_tp, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx,
        const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars);

    void elwise_resolve_option_values(
        const arrfunc_type_data *self, const arrfunc_type *self_tp,
        intptr_t nsrc, const ndt::type *src_tp, nd::array &kwds,
        const std::map<nd::string, ndt::type> &tp_vars);

    int elwise_resolve_dst_type(
        const arrfunc_type_data *self, const arrfunc_type *DYND_UNUSED(self_tp),
        intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
        ndt::type &dst_tp, const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars);

    int elwise_resolve_dst_type_with_child(
        const arrfunc_type_data *child, const arrfunc_type *child_tp,
        intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
        ndt::type &dst_tp, const dynd::nd::array &kwds,
        const std::map<dynd::nd::string, ndt::type> &tp_vars);

    ndt::type elwise_make_type(const arrfunc_type *child_tp);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
