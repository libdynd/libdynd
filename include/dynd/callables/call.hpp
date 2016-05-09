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
    void (*destroy)(call_node *);
    void (*instantiate)(call_node *, kernel_builder *, kernel_request_t, char *, const char *, size_t,
                        const char *const *);
    size_t data_size;

    template <typename ClosureType, typename... ArgTypes>
    static void init(call_node *self, ArgTypes &&... args) {
      new (self) call_node{[](call_node *self) {
                             ClosureType *closure = reinterpret_cast<ClosureType *>(reinterpret_cast<char *>(self) +
                                                                                    aligned_size(sizeof(call_node)));
                             closure->~ClosureType();
                           },
                           [](call_node *self, kernel_builder *kb, kernel_request_t kernreq, char *data,
                              const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
                             ClosureType *closure = reinterpret_cast<ClosureType *>(reinterpret_cast<char *>(self) +
                                                                                    aligned_size(sizeof(call_node)));
                             (*closure)(*kb, kernreq, data, dst_arrmeta, nsrc, src_arrmeta);
                           },
                           aligned_size(sizeof(call_node)) + aligned_size(sizeof(ClosureType))};

      new (reinterpret_cast<char *>(self) + aligned_size(sizeof(call_node)))
          ClosureType(std::forward<ArgTypes>(args)...);
    }
  };

} // namespace dynd::nd
} // namespace dynd
