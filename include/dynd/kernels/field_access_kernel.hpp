//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/func/assignment.hpp>

namespace dynd {
namespace nd {

  struct field_access_kernel : base_strided_kernel<field_access_kernel, 1> {
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
      return s->get_field_type(name);
    }

    static uintptr_t get_data_offset(const ndt::type *src_tp, const char *const *src_arrmeta, const array *kwds)
    {
      const ndt::struct_type *s = src_tp->extended<ndt::struct_type>();
      const std::string &name = kwds[0].as<std::string>();
      return s->get_data_offset(src_arrmeta[0], name);
    }
  };

  // Temporary solution (previously in struct_type.cpp).
  struct get_array_field_kernel : nd::base_kernel<get_array_field_kernel> {
    array self;
    intptr_t i;

    get_array_field_kernel(const array &self, intptr_t i) : self(self), i(i) {}

    void call(array *dst, const array *DYND_UNUSED(src))
    {
      array res = helper(self, i);
      *dst = res;
    }

    static void resolve_dst_type(char *static_data, char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const array *kwds,
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = helper(kwds[0], *reinterpret_cast<intptr_t *>(static_data)).get_type();
    }

    static void instantiate(char *static_data, char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                            const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                            intptr_t DYND_UNUSED(nkwd), const array *kwds,
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      ckb->emplace_back<get_array_field_kernel>(kernreq, kwds[0], *reinterpret_cast<intptr_t *>(static_data));
    }

    static array helper(const array &n, intptr_t i)
    {
      // Get the nd::array 'self' parameter
      intptr_t undim = n.get_ndim();
      ndt::type udt = n.get_dtype();
      if (udt.is_expression()) {
        std::string field_name = udt.value_type().extended<ndt::struct_type>()->get_field_name(i);
        return n.replace_dtype(ndt::make_type<ndt::adapt_type>(
            udt.value_type().extended<ndt::struct_type>()->get_field_type(i), udt, nd::callable(), nd::callable()));
      }
      else {
        if (undim == 0) {
          return n(i);
        }
        else {
          shortvector<irange> idx(undim + 1);
          idx[undim] = irange(i);
          return n.at_array(undim + 1, idx.get());
        }
      }
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
