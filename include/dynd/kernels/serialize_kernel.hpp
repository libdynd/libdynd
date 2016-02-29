//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  struct serialize_kernel;

  template <>
  struct serialize_kernel<scalar_kind_id> : base_strided_kernel<serialize_kernel<scalar_kind_id>, 1> {
    size_t data_size;

    serialize_kernel(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src) { reinterpret_cast<bytes *>(dst)->append(src[0], data_size); }

    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                            const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                            intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      ckb->emplace_back<serialize_kernel>(kernreq, src_tp[0].get_data_size());
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Arg0ID>
  struct traits<nd::serialize_kernel<Arg0ID>> {
    static type equivalent() { return callable_type::make(type("bytes"), {type(Arg0ID)}); }
  };

} // namespace dynd::ndt
} // namespace dynd
