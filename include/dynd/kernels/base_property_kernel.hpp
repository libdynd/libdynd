//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename SelfType>
  struct base_property_kernel : base_kernel<SelfType, 0> {
    typedef SelfType self_type;

    static const std::size_t data_size = sizeof(ndt::type);

    ndt::type tp;
    const ndt::type &dst_tp;
    const char *dst_arrmeta;

    base_property_kernel(const ndt::type &tp, const ndt::type &dst_tp, const char *dst_arrmeta)
        : tp(tp), dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
    {
    }

    static char *data_init(char *DYND_UNUSED(static_data), const ndt::type &DYND_UNUSED(dst_tp),
                           intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd),
                           const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return reinterpret_cast<char *>(new ndt::type(kwds[0].as<ndt::type>()));
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                                const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
                                intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      self_type::make(ckb, kernreq, ckb_offset, *reinterpret_cast<ndt::type *>(data), dst_tp, dst_arrmeta);
      delete reinterpret_cast<ndt::type *>(data);
      return ckb_offset;
    }
  };

} // namespace dynd::nd
} // namespace dynd
