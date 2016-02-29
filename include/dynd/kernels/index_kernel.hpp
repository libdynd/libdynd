//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_index_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  struct index_kernel : base_index_kernel<index_kernel<Arg0ID>> {
    using base_index_kernel<index_kernel>::base_index_kernel;

    typedef typename base_index_kernel<index_kernel<Arg0ID>>::data_type data_type;

    void single(char *DYND_UNUSED(metadata), char **DYND_UNUSED(data), const array *DYND_UNUSED(arg0)) {}

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *data, ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd),
                                 const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = src_tp[0];
      new (data) data_type(kwds[0]);
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                            const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                            intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      ckb->emplace_back<index_kernel>(kernreq);
      delete reinterpret_cast<typename base_index_kernel<index_kernel<Arg0ID>>::data_type *>(data);
    }
  };

  template <>
  struct index_kernel<fixed_dim_id> : base_index_kernel<index_kernel<fixed_dim_id>> {
    intptr_t index;
    intptr_t stride;

    index_kernel(int index, intptr_t stride) : index(index), stride(stride) {}

    ~index_kernel() { get_child()->destroy(); }

    void single(char *metadata, char **data, const array *DYND_UNUSED(arg0))
    {
      //      reinterpret_cast<ndt::fixed_dim_type::metadata_type *>(metadata)->stride = stride;
      *data += index * stride;

      get_child()->single(metadata, data, NULL);
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *data, ndt::type &dst_tp, intptr_t nsrc,
                                 const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                                 const std::map<std::string, ndt::type> &tp_vars)
    {
      reinterpret_cast<data_type *>(data)->next();

      ndt::type child_src_tp = src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
      index::get()->resolve_dst_type(index::get()->static_data(), data, dst_tp, nsrc, &child_src_tp, nkwd, kwds,
                                     tp_vars);
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                            const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                            const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                            const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      ckb->emplace_back<index_kernel>(
          kernreq, *reinterpret_cast<data_type *>(data)->indices,
          reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(src_arrmeta[0])->stride);

      reinterpret_cast<data_type *>(data)->next();

      ndt::type child_src_tp = src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
      const char *child_src_arrmeta = src_arrmeta[0] + sizeof(ndt::fixed_dim_type::metadata_type);
      index::get()->instantiate(index::get()->static_data(), data, ckb, dst_tp, dst_arrmeta, nsrc, &child_src_tp,
                                &child_src_arrmeta, kernel_request_single, nkwd, kwds, tp_vars);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t ResID>
  struct traits<nd::index_kernel<ResID>> {
    static type equivalent() { return type("(Any, i: Any) -> Any"); }
  };

} // namespace dynd::ndt
} // namespace dynd
