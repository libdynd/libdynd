//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/assign_na_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename ResValueType>
  class assign_na_callable : public default_instantiable_callable<assign_na_kernel<ResValueType>> {
  public:
    assign_na_callable()
        : default_instantiable_callable<assign_na_kernel<ResValueType>>(
              ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::option_type>(ndt::make_type<ResValueType>()))) {}
  };

  template <>
  class assign_na_callable<ndt::fixed_dim_type> : public base_callable {
  public:
    assign_na_callable()
        : base_callable(ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::option_type>(fixed_dim_id))) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      switch (dst_tp.get_dtype().get_id()) {
      case bool_id:
        cg.emplace_back(
            [](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
               const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
               const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<assign_na_kernel<bool>>(kernreq); });
        break;
      case int8_id:
        cg.emplace_back(
            [](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
               const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
               const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<assign_na_kernel<int8_t>>(kernreq); });
        break;
      case int16_id:
        cg.emplace_back(
            [](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
               const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
               const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<assign_na_kernel<int16_t>>(kernreq); });
        break;
      case int32_id:
        cg.emplace_back(
            [](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
               const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
               const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<assign_na_kernel<int32_t>>(kernreq); });
        break;
      case int64_id:
        cg.emplace_back(
            [](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
               const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
               const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<assign_na_kernel<int64_t>>(kernreq); });
        break;
      case int128_id:
        cg.emplace_back(
            [](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
               const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
               const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<assign_na_kernel<int128>>(kernreq); });
        break;
      case float32_id:
        cg.emplace_back(
            [](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
               const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
               const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<assign_na_kernel<float>>(kernreq); });
        break;
      case float64_id:
        cg.emplace_back(
            [](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
               const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
               const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<assign_na_kernel<double>>(kernreq); });
        break;
      case complex_float32_id:
        cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                           const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
                           const char *const *DYND_UNUSED(src_arrmeta)) {
          kb.emplace_back<assign_na_kernel<complex<float>>>(kernreq);
        });
        break;
      case complex_float64_id:
        cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                           const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
                           const char *const *DYND_UNUSED(src_arrmeta)) {
          kb.emplace_back<assign_na_kernel<complex<double>>>(kernreq);
        });
        break;
      default:
        throw type_error("fixed_dim_assign_na: expected built-in type");
      }

      return dst_tp;
    }
  };

} // namespace dynd::nd
} // namespace dynd
