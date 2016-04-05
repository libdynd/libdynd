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

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(res_tp), size_t DYND_UNUSED(narg), const ndt::type *arg_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      std::string name = kwds[0].as<std::string>();
      intptr_t i = arg_tp[0].extended<ndt::struct_type>()->get_field_index(name);
      uintptr_t arrmeta_offset = arg_tp[0].extended<ndt::struct_type>()->get_arrmeta_offset(i);

      cg.push_back([i, arrmeta_offset](kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                                       size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        uintptr_t data_offset = *(reinterpret_cast<const uintptr_t *>(src_arrmeta[0]) + i);
        const char *field_metadata[1] = {src_arrmeta[0] + arrmeta_offset};

        ckb->emplace_back<field_access_kernel>(kernreq, data_offset);

        ckb->instantiate(kernreq | kernel_request_data_only, dst_arrmeta, 1, field_metadata);
      });

      ndt::type field_tp = arg_tp[0].extended<ndt::struct_type>()->get_field_type(i);
      assign->resolve(this, nullptr, cg, field_tp, 1, &field_tp, 1, nullptr, tp_vars);

      return field_tp;
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
      cg.push_back([ kwds, m_i = m_i ](kernel_builder * ckb, kernel_request_t kernreq,
                                       const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
                                       const char *const *DYND_UNUSED(src_arrmeta)) {
        ckb->emplace_back<get_array_field_kernel>(kernreq, kwds[0], m_i);
      });

      return get_array_field_kernel::helper(kwds[0], m_i).get_type();
    }
  };

} // namespace dynd::nd
} // namespace dynd
