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
      intptr_t nindices;
      int *indices;

      data_type(const array &index) : nindices(index.get_dim_size()), indices(reinterpret_cast<int *>(index.data())) {}
    };

    int index;

    base_index_kernel(int index) : index(index) {}

    void call(array *res, array *const *args)
    {
      res->get()->data = args[0]->get()->data;
      reinterpret_cast<SelfType *>(this)->single(res->get()->metadata(), &res->get()->data);
    }

    static char *data_init(char *DYND_UNUSED(static_data), const ndt::type &DYND_UNUSED(dst_tp),
                           intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd),
                           const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return reinterpret_cast<char *>(new data_type(kwds[0]));
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      SelfType::make(ckb, kernreq, ckb_offset, *reinterpret_cast<data_type *>(data)->indices);
      delete reinterpret_cast<data_type *>(data);

      return ckb_offset;
    }
  };

  template <type_id_t Arg0ID>
  struct index_kernel : base_index_kernel<index_kernel<Arg0ID>> {
    using base_index_kernel<index_kernel>::base_index_kernel;
  };

  template <>
  struct index_kernel<fixed_dim_type_id> : base_index_kernel<index_kernel<fixed_dim_type_id>> {
    intptr_t stride;

    index_kernel(int index, intptr_t stride) : base_index_kernel(index), stride(stride) {}

    void single(char *metadata, char *const *data)
    {
      reinterpret_cast<ndt::fixed_dim_type::metadata_type *>(metadata)->stride = stride;
      *const_cast<char **>(data) += index * stride;
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *data, ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd),
                                 const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      if (reinterpret_cast<data_type *>(data)->nindices == 1) {
        dst_tp = src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
      }
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                                const nd::array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset, *reinterpret_cast<data_type *>(data)->indices,
           reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(src_arrmeta[0])->stride);
      delete reinterpret_cast<data_type *>(data);

      return ckb_offset;
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
