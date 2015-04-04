//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/copy.hpp>

using namespace std;
using namespace dynd;

intptr_t nd::copy_ck::instantiate(
    const arrfunc_type_data *self, const arrfunc_type *af_tp,
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  if (dst_tp.is_builtin()) {
    if (src_tp[0].is_builtin()) {
      if (dst_tp.extended() == src_tp[0].extended()) {
        return make_pod_typed_data_assignment_kernel(
            ckb, ckb_offset, dynd::detail::builtin_data_sizes
                                 [dst_tp.unchecked_get_builtin_type_id()],
            dynd::detail::builtin_data_alignments
                [dst_tp.unchecked_get_builtin_type_id()],
            kernreq);
      } else {
        return make_builtin_type_assignment_kernel(
            ckb, ckb_offset, dst_tp.get_type_id(), src_tp[0].get_type_id(),
            kernreq, ectx->errmode);
      }
    } else {
      return src_tp[0].extended()->make_assignment_kernel(
          self, af_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0],
          src_arrmeta[0], kernreq, ectx, kwds);
    }
  } else {
    return dst_tp.extended()->make_assignment_kernel(
        self, af_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0],
        src_arrmeta[0], kernreq, ectx, kwds);
  }
}

void nd::copy_ck::resolve_dst_type(
    const arrfunc_type_data *DYND_UNUSED(self),
    const arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
    ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
    const nd::array &DYND_UNUSED(kwds),
    const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  if (nsrc != 1) {
    std::stringstream ss;
    ss << "arrfunc 'copy' expected 1 argument, got " << nsrc;
    throw std::invalid_argument(ss.str());
  }

  dst_tp = src_tp[0].get_canonical_type();
}