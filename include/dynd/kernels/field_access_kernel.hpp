//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/func/assignment.hpp>

namespace dynd {
namespace nd {

  struct field_access_kernel : base_kernel<field_access_kernel, 1> {
    const uintptr_t data_offset;

    field_access_kernel(uintptr_t data_offset) : data_offset(data_offset) {}

    ~field_access_kernel() { get_child()->destroy(); }

    void single(char *res, char *const *src)
    {
      char *const field_src[1] = {src[0] + data_offset};
      get_child()->single(res, field_src);
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd),
                                 const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = get_field_type(src_tp, kwds);
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                            const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                            const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                            const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      const uintptr_t data_offset = get_data_offset(src_tp, src_arrmeta, kwds);
      const ndt::type field_type[1] = {get_field_type(src_tp, kwds)};
      const nd::array field_value = nd::empty(field_type[0]);
      const char *field_metadata[1] = {field_value->metadata()};

      ckb->emplace_back<field_access_kernel>(kernreq, data_offset);

      static const array error_mode(opt<assign_error_mode>());
      assign::get()->instantiate(assign::get()->static_data(), data, ckb, dst_tp, dst_arrmeta, 1, field_type,
                                 field_metadata, kernreq | kernel_request_data_only, 1, &error_mode, tp_vars);
    };

    static const ndt::type &get_field_type(const ndt::type *src_tp, const array *kwds)
    {
      const ndt::struct_type *s = src_tp->extended<ndt::struct_type>();
      const std::string &name = kwds[0].as<std::string>();
      uintptr_t index = s->get_field_index(name);
      return s->get_field_type(index);
    }

    static uintptr_t get_data_offset(const ndt::type *src_tp, const char *const *src_arrmeta, const array *kwds)
    {
      const ndt::struct_type *s = src_tp->extended<ndt::struct_type>();
      const std::string &name = kwds[0].as<std::string>();
      uintptr_t index = s->get_field_index(name);
      return s->get_data_offsets(src_arrmeta[0])[index];
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct traits<nd::field_access_kernel> {
    static type equivalent() { return type("({...}, field_name : string) -> Any"); }
  };

} // namespace dynd::ndt

} // namespace dynd
