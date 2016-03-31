//
// Copyright (C) 2011-15 DyND Developers
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

    void resolve(call_graph &cg) { cg.emplace_back(this); }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<byteswap_ck>(kernreq, src_tp[0].get_data_size());
    }

    /*
        virtual void new_instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                     const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const array
       *DYND_UNUSED(kwds))
        {
          ckb->emplace_back<byteswap_ck>(kernreq, src_tp[0].get_data_size());
        }
    */
  };

  class pairwise_byteswap_callable : public base_callable {
  public:
    pairwise_byteswap_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    void resolve(call_graph &cg) { cg.emplace_back(this); }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<pairwise_byteswap_ck>(kernreq, src_tp[0].get_data_size());
    }

    /*
        virtual void new_instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                     const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const array
       *DYND_UNUSED(kwds))
        {
          ckb->emplace_back<pairwise_byteswap_ck>(kernreq, src_tp[0].get_data_size());
        }
    */
  };

} // namespace dynd::nd
} // namespace dynd
