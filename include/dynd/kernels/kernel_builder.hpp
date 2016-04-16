//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/call.hpp>
#include <dynd/storagebuf.hpp>

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
    call_node *m_call;

  public:
    kernel_builder(call_node *call = nullptr) : m_call(call) {}

    DYND_API void destroy();

    ~kernel_builder() { destroy(); }

    template <typename KernelType, typename... ArgTypes>
    void emplace_back(ArgTypes &&... args) {
      storagebuf<kernel_prefix, kernel_builder>::emplace_back<KernelType>(std::forward<ArgTypes>(args)...);

      m_call = reinterpret_cast<call_node *>(reinterpret_cast<char *>(m_call) + m_call->aligned_size);
    }

    void emplace_back(size_t size) { storagebuf<kernel_prefix, kernel_builder>::emplace_back(size); }

    void pass() { m_call = reinterpret_cast<call_node *>(reinterpret_cast<char *>(m_call) + m_call->aligned_size); }

    void operator()(kernel_request_t kr, char *data, const char *res_metadata, size_t narg,
                    const char *const *arg_metadata) {
      m_call->instantiate(m_call, this, kr, data, res_metadata, narg, arg_metadata);
    }
  };

} // namespace dynd::nd
} // namespace dynd
