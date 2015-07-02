#pragma once

#include <dynd/kernels/base_virtual_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename T>
  struct call_kernel : base_virtual_kernel<call_kernel<T>> {
    static void data_init(const arrfunc_type_data *DYND_UNUSED(self),
                          const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                          const char *static_data, size_t data_size, char *data,
                          intptr_t nsrc, const ndt::type *src_tp,
                          dynd::nd::array &kwds,
                          const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &func = T::get_self();

      func.get()->data_init(func.get(), func.get_type(), static_data, data_size,
                            data, nsrc, src_tp, kwds, tp_vars);
    }

    static void
    resolve_dst_type(const arrfunc_type_data *DYND_UNUSED(self),
                     const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                     const char *static_data, size_t data_size, char *data,
                     ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                     const dynd::nd::array &kwds,
                     const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &func = T::get_self();

      func.get()->resolve_dst_type(func.get(), func.get_type(), static_data,
                                   data_size, data, dst_tp, nsrc, src_tp, kwds,
                                   tp_vars);
    }

    static intptr_t
    instantiate(const arrfunc_type_data *DYND_UNUSED(self),
                const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                const char *static_data, size_t data_size, char *data,
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      const arrfunc &func = T::get_self();

      return func.get()->instantiate(func.get(), func.get_type(), static_data,
                                     data_size, data, ckb, ckb_offset, dst_tp,
                                     dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                     kernreq, ectx, kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd