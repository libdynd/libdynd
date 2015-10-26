//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/bytes.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct sort_kernel : base_kernel<sort_kernel, 1> {
    class iterator;

    static const size_t data_size = 0;

    const intptr_t src0_size;
    const intptr_t src0_stride;
    const intptr_t src0_element_data_size;

    sort_kernel(intptr_t src0_size, intptr_t src0_stride, size_t src0_element_data_size)
        : src0_size(src0_size), src0_stride(src0_stride), src0_element_data_size(src0_element_data_size)
    {
    }

    void single(char *DYND_UNUSED(dst), char *const *src)
    {
      std::sort(bytes_iterator(src[0], src0_element_data_size, src0_stride),
                bytes_iterator(src[0] + src0_size * src0_stride, src0_element_data_size, src0_stride),
                [](const std_bytes &lhs, const std_bytes &rhs) {
        return *reinterpret_cast<const int *>(lhs.data()) < *reinterpret_cast<const int *>(rhs.data());
      });
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                                 ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                 intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = src_tp[0];
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                                void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                const nd::array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
           reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride,
           src_tp[0].template extended<ndt::fixed_dim_type>()->get_element_type().get_data_size());

      return ckb_offset;
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::sort_kernel> {
    static type make()
    {
      return callable_type::make(type("Fixed * Any"), {type("Fixed * Any")});
    }
  };

} // namespace dynd::ndt
} // namespace dynd
