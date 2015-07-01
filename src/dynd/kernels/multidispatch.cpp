//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/buffered_kernels.hpp>
#include <dynd/kernels/multidispatch.hpp>

using namespace std;
using namespace dynd;

intptr_t nd::functional::old_multidispatch_ck::instantiate(
    const arrfunc_type_data *af_self,
    const ndt::arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  const vector<nd::arrfunc> *icd = af_self->get_data_as<vector<nd::arrfunc>>();
  for (intptr_t i = 0; i < (intptr_t)icd->size(); ++i) {
    const nd::arrfunc &af = (*icd)[i];
    intptr_t isrc, nsrc = af.get_type()->get_npos();
    std::map<nd::string, ndt::type> typevars;
    for (isrc = 0; isrc < nsrc; ++isrc) {
      if (!can_implicitly_convert(
              src_tp[isrc], af.get_type()->get_pos_type(isrc), typevars)) {
        break;
      }
    }
    if (isrc == nsrc) {
      intptr_t j;
      for (j = 0; j < nsrc; ++j) {
        const ndt::type &arg_tp = af.get_type()->get_pos_type(j);
        if (!arg_tp.is_symbolic() && src_tp[j] != arg_tp) {
          break;
        }
      }
      if (j == nsrc) {
        return af.get()->instantiate(
            af.get(), af.get_type(), NULL, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      } else {
        return make_buffered_ckernel(af.get(), af.get_type(), ckb, ckb_offset,
                                     dst_tp, dst_arrmeta, nsrc, src_tp,
                                     af.get_type()->get_pos_types_raw(),
                                     src_arrmeta, kernreq, ectx);
      }
    }
  }
  // TODO: Good message here
  stringstream ss;
  ss << "No matching signature found in multidispatch arrfunc";
  throw invalid_argument(ss.str());
}

void nd::functional::old_multidispatch_ck::resolve_dst_type(
    const arrfunc_type_data *self,
    const ndt::arrfunc_type *DYND_UNUSED(self_tp), size_t data_size, char *data,
    ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
{
  const vector<nd::arrfunc> *icd = self->get_data_as<vector<nd::arrfunc>>();
  for (intptr_t i = 0; i < (intptr_t)icd->size(); ++i) {
    const nd::arrfunc &child = (*icd)[i];
    if (nsrc == child.get_type()->get_npos()) {
      intptr_t isrc;
      std::map<nd::string, ndt::type> typevars;
      for (isrc = 0; isrc < nsrc; ++isrc) {
        if (!can_implicitly_convert(
                src_tp[isrc], child.get_type()->get_pos_type(isrc), typevars)) {
          break;
        }
      }
      if (isrc == nsrc) {
        child.get()->resolve_dst_type(child.get(), child.get_type(), data_size,
                                      data, dst_tp, nsrc, src_tp, kwds,
                                      tp_vars);
        return;
      }
    }
  }

  stringstream ss;
  ss << "Failed to find suitable signature in multidispatch resolution "
        "with "
        "input types (";
  for (intptr_t isrc = 0; isrc < nsrc; ++isrc) {
    ss << src_tp[isrc];
    if (isrc != nsrc - 1) {
      ss << ", ";
    }
  }
  ss << ")";
  throw type_error(ss.str());
}