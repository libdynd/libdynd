//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/call.hpp>
#include <dynd/callables/closure_call.hpp>
#include <dynd/storagebuf.hpp>

namespace dynd {
namespace nd {

  class call_graph : public storagebuf<call_node, call_graph> {
  public:
    void destroy() {}

    ~call_graph() {
      intptr_t offset = 0;
      while (offset != m_size) {
        call_node *node = get_at<call_node>(offset);
        offset += node->aligned_size;
        node->destroy(node);
      }
    }

    template <typename ClosureType>
    void emplace_back(ClosureType closure) {
      storagebuf<call_node, call_graph>::emplace_back<closure_call<ClosureType>>(closure);
    }
  };

} // namespace dynd::nd
} // namespace dynd
