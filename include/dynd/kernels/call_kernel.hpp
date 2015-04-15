#pragma once

#include <dynd/kernels/base_virtual_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename T>
  struct call_kernel : base_virtual_kernel<call_kernel<T>> {
    static intptr_t
    instantiate(const arrfunc_type_data *self,
                const arrfunc_type *DYND_UNUSED(self_tp), char *data, void *ckb,
                intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      const T *func = *self->get_data_as<T *>();

      return func->get()->instantiate(
          func->get(), func->get_type(), data, ckb, ckb_offset, dst_tp,
          dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

    static void
    resolve_dst_type(const arrfunc_type_data *self,
                     const arrfunc_type *DYND_UNUSED(self_tp), char *data,
                     ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                     const dynd::nd::array &kwds,
                     const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const T *func = *self->get_data_as<T *>();

      func->get()->resolve_dst_type(func->get(), func->get_type(), data, dst_tp,
                                    nsrc, src_tp, kwds, tp_vars);
    }

    static void
    resolve_option_values(const arrfunc_type_data *self,
                          const arrfunc_type *DYND_UNUSED(self_tp), char *data,
                          intptr_t nsrc, const ndt::type *src_tp,
                          dynd::nd::array &kwds,
                          const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const T *func = *self->get_data_as<T *>();

      func->get()->resolve_option_values(func->get(), func->get_type(), data,
                                         nsrc, src_tp, kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd