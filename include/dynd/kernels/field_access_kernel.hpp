//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct field_access_kernel : base_kernel<field_access_kernel, 1> {
    const uintptr_t data_offset;
    const size_t data_size;

    field_access_kernel(uintptr_t data_offset, size_t data_size) : data_offset(data_offset), data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      memcpy(dst, src+data_offset, data_size);
    }

    static void instantiate(char *DYND_UNUSED(static_data),
                            char *DYND_UNUSED(data),
                            kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp),
                            const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc),
                            const ndt::type *src_tp,
                            const char *const *src_arrmeta,
                            kernel_request_t kernreq,
                            intptr_t DYND_UNUSED(nkwd),
                            const array *kwds,
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const ndt::struct_type *s = src_tp->extended<ndt::struct_type>();
      const std::string &name = kwds[0].as<std::string>();
      uintptr_t i = s->get_field_index(name);
      uintptr_t data_offset = reinterpret_cast<const uintptr_t *>(src_arrmeta[0])[i];
      size_t data_size = src_tp->get_data_size();

      ckb->emplace_back<field_access_kernel>(kernreq, data_offset, data_size);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct traits<nd::field_access_kernel> {
    static type equivalent() { return type("({...}, field_name : string) -> void"); }
  };

} // namespace dynd::ndt

} // namespace dynd
