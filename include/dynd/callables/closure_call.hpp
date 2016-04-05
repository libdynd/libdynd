//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/call.hpp>

namespace dynd {
namespace nd {

  template <typename ClosureType>
  struct closure_call : call_node {
    ClosureType closure;

    closure_call(ClosureType closure)
        : call_node(instantiate_wrapper, destructor_wrapper, sizeof(closure_call)), closure(closure) {}

    template <typename... ArgTypes>
    static void init(closure_call *self, ArgTypes &&... args) {
      new (self) closure_call(std::forward<ArgTypes>(args)...);
    }

    static void destructor_wrapper(call_node *self) { reinterpret_cast<closure_call *>(self)->~closure_call(); }

    static void instantiate_wrapper(call_node *&self, kernel_builder *ckb, kernel_request_t kernreq,
                                    const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
      reinterpret_cast<closure_call *>(self)->closure(self, ckb, kernreq, dst_arrmeta, nsrc, src_arrmeta);
    }
  };

} // namespace dynd::nd
} // namespace dynd
