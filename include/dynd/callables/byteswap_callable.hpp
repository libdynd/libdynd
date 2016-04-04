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
    struct node_type : call_node {
      size_t data_size;

      node_type(base_callable *callee, size_t data_size) : call_node(callee), data_size(data_size) {}
    };

  public:
    byteswap_callable() : base_callable(ndt::type("(Any) -> Any"), sizeof(node_type)) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back<node_type>(this, src_tp[0].get_data_size());
      return dst_tp;
    }

    void instantiate(call_node *&node, char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                     const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<byteswap_ck>(kernreq, reinterpret_cast<node_type *>(node)->data_size);
      node = next(node);
    }
  };

  class pairwise_byteswap_callable : public base_callable {
  public:
    pairwise_byteswap_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(call_node *&node, char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<pairwise_byteswap_ck>(kernreq, src_tp[0].get_data_size());
      node = next(node);
    }
  };

} // namespace dynd::nd
} // namespace dynd
