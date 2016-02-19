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
  struct serialize_kernel<scalar_kind_id> : base_kernel<serialize_kernel<scalar_kind_id>, 1> {
    void single(char *dst, char *const *src) { reinterpret_cast<bytes *>(dst)->append(src[0], sizeof(int32_t)); }

    void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride), char *const *DYND_UNUSED(src),
                 const intptr_t *DYND_UNUSED(src_stride), size_t DYND_UNUSED(count))
    {
      std::cout << "serialize_kernel::strided" << std::endl;
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
