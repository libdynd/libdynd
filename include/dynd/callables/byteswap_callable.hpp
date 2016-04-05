//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/byteswap_kernels.hpp>

namespace dynd {
namespace nd {

  class byteswap_callable : public base_callable {
  public:
    byteswap_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      size_t src0_data_size = src_tp[0].get_data_size();
      cg.emplace_back([src0_data_size](
          kernel_builder &kb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
          const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<byteswap_ck>(kernreq, src0_data_size); });

      return dst_tp;
    }
  };

  class pairwise_byteswap_callable : public base_callable {
  public:
    pairwise_byteswap_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      size_t src0_data_size = src_tp[0].get_data_size();
      cg.emplace_back([src0_data_size](kernel_builder &kb, kernel_request_t kernreq,
                                       const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
                                       const char *const *DYND_UNUSED(src_arrmeta)) {
        kb.emplace_back<pairwise_byteswap_ck>(kernreq, src0_data_size);
      });

      return dst_tp;
    }
  };

} // namespace dynd::nd
} // namespace dynd
