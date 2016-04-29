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

      cg.emplace_back([i, arrmeta_offset](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                          const char *dst_arrmeta, size_t DYND_UNUSED(nsrc),
                                          const char *const *src_arrmeta) {
        uintptr_t data_offset = *(reinterpret_cast<const uintptr_t *>(src_arrmeta[0]) + i);
        const char *field_metadata[1] = {src_arrmeta[0] + arrmeta_offset};

        kb.emplace_back<field_access_kernel>(kernreq, data_offset);

        kb(kernreq | kernel_request_data_only, nullptr, dst_arrmeta, 1, field_metadata);
      });

      ndt::type field_tp = arg_tp[0].extended<ndt::struct_type>()->get_field_type(i);
      assign->resolve(this, nullptr, cg, field_tp, 1, &field_tp, 1, nullptr, tp_vars);

      return field_tp;
    }
  };

  class get_array_field_callable : public base_callable {
  public:
    get_array_field_callable()
        : base_callable(ndt::make_type<ndt::callable_type>(ndt::type("Any"), {ndt::type("Any")},
                                                           {{ndt::make_type<std::string>(), "name"}})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(res_tp), size_t DYND_UNUSED(narg), const ndt::type *arg_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ndt::type dt = arg_tp[0].get_dtype();
      std::string name = kwds[0].as<std::string>();

      if (dt.get_id() != struct_id) {
        throw std::invalid_argument(std::string("no property named '") + name + "'");
      }

      intptr_t i = dt.extended<ndt::struct_type>()->get_field_index(name);
      if (i < 0) {
        throw std::invalid_argument("no field named '" + name + "'");
      }

      cg.emplace_back(
          [i](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
              const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
              const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<get_array_field_kernel>(kernreq, i); });

      return arg_tp[0].with_replaced_dtype(dt);
    }
  };

} // namespace dynd::nd
} // namespace dynd
