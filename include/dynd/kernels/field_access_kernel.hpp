//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  // This is an example of accessing fields in a struct. It currently assumes that the struct has types {int, double}.
  struct field_access_kernel : base_kernel<field_access_kernel, 1> {
    static const kernel_request_t kernreq = kernel_request_call;

    const uintptr_t *data_offsets;

    field_access_kernel(const uintptr_t *data_offsets) : data_offsets(data_offsets) {}

    void single(char *DYND_UNUSED(dst), char *const *src)
    {
      std::cout << "x = " << *reinterpret_cast<int *>(src[0] + data_offsets[0]) << std::endl;
      std::cout << "y = " << *reinterpret_cast<double *>(src[0] + data_offsets[1]) << std::endl;
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                            const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                            const array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      ckb->emplace_back<field_access_kernel>(kernreq, reinterpret_cast<const uintptr_t *>(src_arrmeta[0]));
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct traits<nd::field_access_kernel> {
    static type equivalent() { return type("({...}) -> void"); }
  };

} // namespace dynd::ndt

} // namespace dynd
