//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct real_kernel : base_kernel<real_kernel> {
    array self;

    real_kernel(const array &self) : self(self) {}

    void single(array *dst, array *const *DYND_UNUSED(src)) { *dst = helper(self); }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const array *kwds,
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = helper(kwds[0]).get_type();
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset, kwds[0]);
      return ckb_offset;
    }

    static array helper(const array &n) { return n.replace_dtype(ndt::property_type::make(n.get_dtype(), "real")); }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::real_kernel> {
    static type make() { return type("(self: Any) -> Any"); }
  };

} // namespace dynd::ndt
} // namespace dynd
