//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/comparison.hpp>
#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/not_equal_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  class not_equal_callable : public base_instantiable_callable<not_equal_kernel<Arg0ID, Arg1ID>> {
  public:
    not_equal_callable()
        : base_instantiable_callable<not_equal_kernel<Arg0ID, Arg1ID>>(
              ndt::callable_type::make(ndt::make_type<bool1>(), {ndt::type(Arg0ID), ndt::type(Arg1ID)}))
    {
    }
  };

  template <>
  class not_equal_callable<tuple_id, tuple_id> : public base_callable {
  public:
    not_equal_callable()
        : base_callable(ndt::callable_type::make(ndt::make_type<bool1>(), {ndt::type(tuple_id), ndt::type(tuple_id)}))
    {
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                     intptr_t nkwd, const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t self_offset = ckb->size();
      auto bsd = src_tp->extended<ndt::tuple_type>();
      size_t field_count = bsd->get_field_count();

      ckb->emplace_back<not_equal_kernel<tuple_id, tuple_id>>(
          kernreq, field_count, bsd->get_data_offsets(src_arrmeta[0]), bsd->get_data_offsets(src_arrmeta[1]));
      ckb->emplace_back(field_count * sizeof(size_t));

      not_equal_kernel<tuple_id, tuple_id> *e = ckb->get_at<not_equal_kernel<tuple_id, tuple_id>>(self_offset);
      size_t *field_kernel_offsets;
      const std::vector<uintptr_t> &arrmeta_offsets = bsd->get_arrmeta_offsets();
      for (size_t i = 0; i != field_count; ++i) {
        const ndt::type &ft = bsd->get_field_type(i);
        // Reserve space for the child, and save the offset to this
        // field comparison kernel. Have to re-get
        // the pointer because creating the field comparison kernel may
        // move the memory.
        e = reinterpret_cast<kernel_builder *>(ckb)->get_at<not_equal_kernel<tuple_id, tuple_id>>(self_offset);
        field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
        field_kernel_offsets[i] = ckb->size() - self_offset;
        const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
        ndt::type child_src_tp[2] = {ft, ft};
        const char *child_src_arrmeta[2] = {field_arrmeta, field_arrmeta};
        not_equal->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, child_src_tp, child_src_arrmeta,
                               kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
