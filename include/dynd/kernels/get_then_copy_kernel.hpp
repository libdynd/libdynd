//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename TypeType, const array &(TypeType::*Func)() const>
  struct get_then_copy_kernel : base_kernel<get_then_copy_kernel<TypeType, Func>, 0> {
    ndt::type tp;

    get_then_copy_kernel(const ndt::type &tp) : tp(tp) {}

    void single(char *res, char *const *DYND_UNUSED(args))
    {
      const array &value = (tp.extended<TypeType>()->*Func)();

      char *child_args = const_cast<char *>(value.cdata());
      this->get_child()->single(res, &child_args);
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                                const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                                kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      const ndt::type &tp = kwds[0].as<ndt::type>();
      get_then_copy_kernel::make(ckb, kernreq, ckb_offset, tp);

      const char *src_metadata = (tp.extended<TypeType>()->*Func)()->metadata();

      static const array error_mode(opt<assign_error_mode>());
      return assign::get()->instantiate(assign::get()->static_data(), data, ckb, ckb_offset, dst_tp, dst_arrmeta, 1,
                                        &dst_tp, &src_metadata, kernreq, ectx, 1, &error_mode, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
