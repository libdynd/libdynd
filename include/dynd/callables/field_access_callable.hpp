//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/field_access_kernel.hpp>

namespace dynd {
namespace nd {

  class field_access_callable : public base_callable {
  public:
    field_access_callable() : base_callable(ndt::type("({...}, field_name : string) -> Any")) {}

    void resolve_dst_type(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                          const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd), const array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      dst_tp = get_field_type(src_tp, kwds);
    }

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(res_tp), size_t DYND_UNUSED(narg), const ndt::type *arg_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return get_field_type(arg_tp, kwds);
    }

    void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      const uintptr_t data_offset = get_data_offset(src_tp, src_arrmeta, kwds);
      const ndt::type field_type[1] = {get_field_type(src_tp, kwds)};
      const nd::array field_value = nd::empty(field_type[0]);
      const char *field_metadata[1] = {field_value->metadata()};

      ckb->emplace_back<field_access_kernel>(kernreq, data_offset);
      node = next(node);

      static const array error_mode(opt<assign_error_mode>());
      node->callee->instantiate(node, data, ckb, dst_tp, dst_arrmeta, 1, field_type, field_metadata,
                                kernreq | kernel_request_data_only, 1, &error_mode, tp_vars);
    };

    const ndt::type &get_field_type(const ndt::type *src_tp, const array *kwds) {
      const ndt::struct_type *s = src_tp->extended<ndt::struct_type>();
      const std::string &name = kwds[0].as<std::string>();
      return s->get_field_type(name);
    }

    uintptr_t get_data_offset(const ndt::type *src_tp, const char *const *src_arrmeta, const array *kwds) {
      const ndt::struct_type *s = src_tp->extended<ndt::struct_type>();
      const std::string &name = kwds[0].as<std::string>();
      return s->get_data_offset(src_arrmeta[0], name);
    }
  };

  class get_array_field_callable : public base_callable {
    intptr_t m_i;

  public:
    get_array_field_callable(intptr_t i)
        : base_callable(
              ndt::callable_type::make(ndt::type("Any"), ndt::tuple_type::make(), ndt::struct_type::make("self"))),
          m_i(i) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(res_tp), size_t DYND_UNUSED(narg),
                      const ndt::type *DYND_UNUSED(arg_tp), size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return get_array_field_kernel::helper(kwds[0], m_i).get_type();
    }

    void resolve_dst_type(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                          const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd), const array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      dst_tp = get_array_field_kernel::helper(kwds[0], m_i).get_type();
    }

    void instantiate(call_node *&node, char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                     const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                     const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<get_array_field_kernel>(kernreq, kwds[0], m_i);
      node = next(node);
    }
  };

} // namespace dynd::nd
} // namespace dynd
