//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename SelfType>
  struct base_index_kernel : base_kernel<SelfType, 1> {
    static const kernel_request_t kernreq = kernel_request_call;

    struct data_type {
      int nindices;
      int *indices;

      data_type(const array &index) : nindices(index.get_dim_size()) {}
    };

    void call(array *DYND_UNUSED(res), array *const *DYND_UNUSED(args))
    {
      // body
    }

    static char *data_init(char *DYND_UNUSED(static_data), const ndt::type &DYND_UNUSED(dst_tp),
                           intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd),
                           const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return reinterpret_cast<char *>(new data_type(kwds[0]));
    }
  };

  template <type_id_t Arg0ID>
  struct index_kernel : base_index_kernel<index_kernel<Arg0ID>> {
  };

  template <>
  struct index_kernel<fixed_dim_type_id> : base_index_kernel<index_kernel<fixed_dim_type_id>> {
    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::type("int32");
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Arg0ID>
  struct traits<nd::index_kernel<Arg0ID>> {
    static type equivalent() { return type("(Any, i: Any) -> Any"); }
  };

} // namespace dynd::ndt
} // namespace dynd
