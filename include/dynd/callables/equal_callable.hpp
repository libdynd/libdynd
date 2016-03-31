//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/comparison.hpp>
#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/equal_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  class equal_callable : public default_instantiable_callable<equal_kernel<Arg0ID, Arg1ID>> {
  public:
    equal_callable()
        : default_instantiable_callable<equal_kernel<Arg0ID, Arg1ID>>(
              ndt::callable_type::make(ndt::make_type<bool1>(), {ndt::type(Arg0ID), ndt::type(Arg1ID)}))
    {
    }
  };

  template <>
  class equal_callable<tuple_id, tuple_id> : public base_callable {
  public:
    equal_callable()
        : base_callable(ndt::callable_type::make(ndt::make_type<bool1>(), {ndt::type(tuple_id), ndt::type(tuple_id)}))
    {
    }

    void resolve(call_graph &cg) { cg.emplace_back(this); }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                     intptr_t nkwd, const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t self_offset = ckb->size();
      size_t field_count = src_tp[0].extended<ndt::tuple_type>()->get_field_count();

      ckb->emplace_back<equal_kernel<tuple_id, tuple_id>>(
          kernreq, field_count, src_tp[0].extended<ndt::tuple_type>()->get_data_offsets(src_arrmeta[0]),
          src_tp[1].extended<ndt::tuple_type>()->get_data_offsets(src_arrmeta[1]));
      ckb->emplace_back(field_count * sizeof(size_t));

      equal_kernel<tuple_id, tuple_id> *self = ckb->get_at<equal_kernel<tuple_id, tuple_id>>(self_offset);
      const std::vector<uintptr_t> &arrmeta_offsets = src_tp[0].extended<ndt::tuple_type>()->get_arrmeta_offsets();
      for (size_t i = 0; i != field_count; ++i) {
        self = ckb->get_at<equal_kernel<tuple_id, tuple_id>>(self_offset);
        size_t *field_kernel_offsets = self->get_offsets();
        field_kernel_offsets[i] = ckb->size() - self_offset;
        ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::tuple_type>()->get_field_type(i),
                                     src_tp[1].extended<ndt::tuple_type>()->get_field_type(i)};
        const char *child_src_arrmeta[2] = {src_arrmeta[0] + arrmeta_offsets[i], src_arrmeta[1] + arrmeta_offsets[i]};
        equal->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, child_src_tp, child_src_arrmeta,
                           kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
