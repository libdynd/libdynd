//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>

namespace dynd {

/**
 * Aligns a size as required by kernels.
 */
static constexpr size_t aligned_size(size_t size) { return (size + static_cast<size_t>(7)) & ~static_cast<size_t>(7); }

typedef uint32_t kernel_request_t;

namespace nd {

  class kernel_builder;

  struct call_node {
    typedef void (*instantiate_type_t)(call_node *node, kernel_builder *ckb, kernel_request_t kernreq, char *data,
                                       const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta);
    typedef void (*destroy_type_t)(call_node *node);

    destroy_type_t destroy;
    instantiate_type_t instantiate;
    size_t aligned_size;

    call_node(instantiate_type_t instantiate)
        : instantiate(instantiate), aligned_size(dynd::aligned_size(sizeof(call_node))) {}

    call_node(instantiate_type_t instantiate, destroy_type_t destroy, size_t data_size)
        : destroy(destroy), instantiate(instantiate), aligned_size(dynd::aligned_size(data_size)) {}

    template <typename... ArgTypes>
    static void init(call_node *self, ArgTypes &&... args) {
      new (self) call_node(std::forward<ArgTypes>(args)...);
    }
  };

} // namespace dynd::nd
} // namespace dynd
