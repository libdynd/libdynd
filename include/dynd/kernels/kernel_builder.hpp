//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/storagebuf.hpp>
#include <dynd/callables/call.hpp>

namespace dynd {
namespace nd {

  struct kernel_prefix;

  /**
   * Function pointers + data for a hierarchical
   * kernel which operates on type/arrmeta in
   * some configuration.
   *
   * The data placed in the kernel's data must
   * be relocatable with a memcpy, it must not rely on its
   * own address.
   */
  class kernel_builder : public storagebuf<kernel_prefix, kernel_builder> {
    call_node *m_node;

  public:
    kernel_builder(call_node *node = nullptr) : m_node(node) {}

    DYND_API void destroy();

    ~kernel_builder() { destroy(); }

    template <typename KernelType, typename... ArgTypes>
    void emplace_back(ArgTypes &&... args) {
      storagebuf<kernel_prefix, kernel_builder>::emplace_back<KernelType>(std::forward<ArgTypes>(args)...);

      if (m_node != nullptr) {
        m_node = reinterpret_cast<call_node *>(reinterpret_cast<char *>(m_node) + m_node->aligned_size);
      }
    }

    void emplace_back(size_t size) { storagebuf<kernel_prefix, kernel_builder>::emplace_back(size); }

    void instantiate(kernel_request_t kernreq, const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
      m_node->instantiate(m_node, this, kernreq, dst_arrmeta, nsrc, src_arrmeta);
    }
  };

} // namespace dynd::nd
} // namespace dynd
