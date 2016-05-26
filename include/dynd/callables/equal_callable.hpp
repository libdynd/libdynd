//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/comparison.hpp>
#include <dynd/kernels/equal_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type, typename Arg1Type>
  class equal_callable : public default_instantiable_callable<equal_kernel<Arg0Type, Arg1Type>> {
  public:
    equal_callable()
        : default_instantiable_callable<equal_kernel<Arg0Type, Arg1Type>>(ndt::make_type<ndt::callable_type>(
              ndt::make_type<bool1>(), {ndt::make_type<Arg0Type>(), ndt::make_type<Arg1Type>()})) {}
  };

  template <>
  class equal_callable<ndt::tuple_type, ndt::tuple_type> : public base_callable {
  public:
    equal_callable()
        : base_callable(ndt::make_type<ndt::callable_type>(
              ndt::make_type<bool1>(), {ndt::make_type<ndt::tuple_type>(), ndt::make_type<ndt::tuple_type>()})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd,
                      const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      size_t field_count = src_tp[0].extended<ndt::tuple_type>()->get_field_count();

      auto bsd = src_tp->extended<ndt::tuple_type>();
      std::array<uintptr_t, 8> arrmeta_offsets;
      for (size_t i = 0; i < field_count; ++i) {
        arrmeta_offsets[i] = bsd->get_arrmeta_offsets()[i];
      }

      cg.emplace_back([field_count, arrmeta_offsets](kernel_builder &kb, kernel_request_t kernreq,
                                                     char *DYND_UNUSED(data), const char *dst_arrmeta, size_t nsrc,
                                                     const char *const *src_arrmeta) {
        intptr_t self_offset = kb.size();

        kb.emplace_back<equal_kernel<ndt::tuple_type, ndt::tuple_type>>(
            kernreq, field_count, reinterpret_cast<const uintptr_t *>(src_arrmeta[0]),
            reinterpret_cast<const uintptr_t *>(src_arrmeta[1]));
        kb.emplace_back(field_count * sizeof(size_t));

        equal_kernel<ndt::tuple_type, ndt::tuple_type> *e =
            kb.get_at<equal_kernel<ndt::tuple_type, ndt::tuple_type>>(self_offset);
        size_t *field_kernel_offsets;
        for (size_t i = 0; i != field_count; ++i) {
          // Reserve space for the child, and save the offset to this
          // field comparison kernel. Have to re-get
          // the pointer because creating the field comparison kernel may
          // move the memory.
          e = kb.get_at<equal_kernel<ndt::tuple_type, ndt::tuple_type>>(self_offset);
          field_kernel_offsets = reinterpret_cast<size_t *>(e + 1);
          field_kernel_offsets[i] = kb.size() - self_offset;
          const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
          const char *child_src_arrmeta[2] = {field_arrmeta, field_arrmeta};
          kb(kernreq | kernel_request_data_only, nullptr, dst_arrmeta, nsrc, child_src_arrmeta);
        }
      });

      for (size_t i = 0; i != field_count; ++i) {
        ndt::type src_field_tp[2] = {src_tp[0].extended<ndt::tuple_type>()->get_field_type(i),
                                     src_tp[1].extended<ndt::tuple_type>()->get_field_type(i)};
        equal->resolve(this, nullptr, cg, dst_tp, 1, src_field_tp, nkwd, kwds, tp_vars);
      }

      return dst_tp;
    }
  };

} // namespace dynd::nd
} // namespace dynd
