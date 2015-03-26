//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/virtual.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {

  template <typename CKT, int N>
  struct multidispatch_by_type_id_ck;

  template <typename CKT>
  struct multidispatch_by_type_id_ck<CKT, 1>
      : virtual_ck<multidispatch_by_type_id_ck<CKT, 1>> {
    static int
    resolve_dst_type(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                     intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
                     ndt::type &out_dst_tp, const dynd::nd::array &kwds,
                     const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &child = CKT::children(src_tp[0].get_type_id());
      return child.get()->resolve_dst_type(self, self_tp, nsrc, src_tp,
                                           throw_on_error, out_dst_tp, kwds,
                                           tp_vars);
    }

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const dynd::nd::array &kwds,
                const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &child = CKT::children(src_tp[0].get_type_id());
      return child.get()->instantiate(self, self_tp, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                      kernreq, ectx, kwds, tp_vars);
    }
  };

  template <typename CKT>
  struct multidispatch_by_type_id_ck<CKT, 2>
      : virtual_ck<multidispatch_by_type_id_ck<CKT, 2>> {
    static int
    resolve_dst_type(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                     intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
                     ndt::type &out_dst_tp, const dynd::nd::array &kwds,
                     const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &child =
          CKT::children(src_tp[0].get_type_id(), src_tp[1].get_type_id());
      return child.get()->resolve_dst_type(self, self_tp, nsrc, src_tp,
                                           throw_on_error, out_dst_tp, kwds,
                                           tp_vars);
    }

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const dynd::nd::array &kwds,
                const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &child =
          CKT::children(src_tp[0].get_type_id(), src_tp[1].get_type_id());
      return child.get()->instantiate(self, self_tp, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                      kernreq, ectx, kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd