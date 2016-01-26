//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/assignment.hpp>
#include <dynd/kernels/copy_kernel.hpp>

using namespace std;
using namespace dynd;

void nd::copy_ck::resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                   intptr_t nsrc, const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd),
                                   const array *DYND_UNUSED(kwds),
                                   const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  if (nsrc != 1) {
    std::stringstream ss;
    ss << "arrfunc 'copy' expected 1 argument, got " << nsrc;
    throw std::invalid_argument(ss.str());
  }

  dst_tp = src_tp[0].get_canonical_type();
}

void nd::copy_ck::instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  if (dst_tp.is_builtin()) {
    if (src_tp[0].is_builtin()) {
      if (dst_tp.extended() == src_tp[0].extended()) {
        make_pod_typed_data_assignment_kernel(
            ckb, dynd::ndt::detail::builtin_data_sizes[dst_tp.unchecked_get_builtin_type_id()],
            dynd::ndt::detail::builtin_data_alignments[dst_tp.unchecked_get_builtin_type_id()], kernreq);
        return;
      }
      else {
        array error_mode = eval::default_eval_context.errmode;
        assign::get()->instantiate(assign::get()->static_data(), NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, 1, src_tp,
                                   src_arrmeta, kernreq, 1, &error_mode, std::map<std::string, ndt::type>());
        return;
      }
    }
    else {
      make_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, &eval::default_eval_context);
      return;
    }
  }
  else {
    make_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, &eval::default_eval_context);
    return;
  }
}
