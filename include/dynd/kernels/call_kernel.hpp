#pragma once

#include <dynd/kernels/base_virtual_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename CallableType>
  struct call_kernel : base_virtual_kernel<call_kernel<CallableType>> {
    static char *data_init(char *DYND_UNUSED(static_data), const ndt::type &dst_tp, intptr_t nsrc,
                           const ndt::type *src_tp, intptr_t nkwd, const nd::array *kwds,
                           const std::map<std::string, ndt::type> &tp_vars)
    {
      return CallableType::get().get()->data_init(CallableType::get().get()->static_data(), dst_tp, nsrc, src_tp, nkwd,
                                                  kwds, tp_vars);
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *data, ndt::type &dst_tp, intptr_t nsrc,
                                 const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                                 const std::map<std::string, ndt::type> &tp_vars)
    {
      CallableType::get().get()->resolve_dst_type(CallableType::get().get()->static_data(), data, dst_tp, nsrc, src_tp,
                                                  nkwd, kwds, tp_vars);
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
    {
      return CallableType::get().get()->instantiate(CallableType::get().get()->static_data(), data, ckb, ckb_offset,
                                                    dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd,
                                                    kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
