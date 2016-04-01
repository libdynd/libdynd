//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/is_na_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ValueID>
  class is_na_callable : public default_instantiable_callable<is_na_kernel<Arg0ValueID>> {
  public:
    is_na_callable()
        : default_instantiable_callable<is_na_kernel<Arg0ValueID>>(
              ndt::callable_type::make(ndt::make_type<bool1>(), ndt::make_type<ndt::option_type>(Arg0ValueID))) {}
  };

  template <>
  class is_na_callable<fixed_dim_id> : public base_callable {
  public:
    is_na_callable()
        : base_callable(
              ndt::callable_type::make(ndt::make_type<bool1>(), ndt::make_type<ndt::option_type>(fixed_dim_kind_id))) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(call_node *DYND_UNUSED(node), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      switch (src_tp->get_dtype().get_id()) {
      case bool_id:
        ckb->emplace_back<is_na_kernel<bool_id>>(kernreq);
        break;
      case int8_id:
        ckb->emplace_back<is_na_kernel<int8_id>>(kernreq);
        break;
      case int16_id:
        ckb->emplace_back<is_na_kernel<int16_id>>(kernreq);
        break;
      case int32_id:
        ckb->emplace_back<is_na_kernel<int32_id>>(kernreq);
        break;
      case int64_id:
        ckb->emplace_back<is_na_kernel<int64_id>>(kernreq);
        break;
      case int128_id:
        ckb->emplace_back<is_na_kernel<int128_id>>(kernreq);
        break;
      case float32_id:
        ckb->emplace_back<is_na_kernel<float32_id>>(kernreq);
        break;
      case float64_id:
        ckb->emplace_back<is_na_kernel<float64_id>>(kernreq);
        break;
      case complex_float32_id:
        ckb->emplace_back<is_na_kernel<complex_float32_id>>(kernreq);
        break;
      case complex_float64_id:
        ckb->emplace_back<is_na_kernel<complex_float64_id>>(kernreq);
        break;
      default:
        throw type_error("fixed_dim_is_avail: expected built-in type");
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
