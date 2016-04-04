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
    struct node_type : call_node {
      intptr_t i;
      uintptr_t arrmeta_offset;

      node_type(base_callable *callee, intptr_t i, uintptr_t arrmeta_offset)
          : call_node(callee), i(i), arrmeta_offset(arrmeta_offset) {}
    };

  public:
    field_access_callable() : base_callable(ndt::type("({...}, field_name : string) -> Any"), sizeof(node_type)) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(res_tp), size_t DYND_UNUSED(narg), const ndt::type *arg_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      std::string name = kwds[0].as<std::string>();
      intptr_t i = arg_tp[0].extended<ndt::struct_type>()->get_field_index(name);

      cg.emplace_back<node_type>(this, i, arg_tp[0].extended<ndt::struct_type>()->get_arrmeta_offset(i));

      ndt::type field_tp = arg_tp[0].extended<ndt::struct_type>()->get_field_type(i);
      assign->resolve(this, nullptr, cg, field_tp, 1, &field_tp, 1, nullptr, tp_vars);

      return field_tp;
    }

    void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &tp_vars) {
      intptr_t i = reinterpret_cast<node_type *>(node)->i;
      uintptr_t data_offset = *(reinterpret_cast<const uintptr_t *>(src_arrmeta[0]) + i);
      const char *field_metadata[1] = {src_arrmeta[0] + reinterpret_cast<node_type *>(node)->arrmeta_offset};

      ckb->emplace_back<field_access_kernel>(kernreq, data_offset);
      node = next(node);

      node->callee->instantiate(node, data, ckb, ndt::type(), dst_arrmeta, 1, nullptr, field_metadata,
                                kernreq | kernel_request_data_only, 0, nullptr, tp_vars);
    };
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
