//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/call.hpp>
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
        offset += node->data_size;
        node->destroy(node);
      }
    }

    template <typename ClosureType, typename... ArgTypes>
    void emplace_back(ArgTypes &&... args) {
      storagebuf<call_node, call_graph>::emplace_back_sep<ClosureType>(std::forward<ArgTypes>(args)...);
    }

    template <typename Arg0Type>
    void emplace_back(Arg0Type &&arg0) {
      typedef remove_reference_then_cv_t<Arg0Type> closure_type;
      this->emplace_back<closure_type, Arg0Type>(std::forward<Arg0Type>(arg0));
    }
  };

} // namespace dynd::nd
} // namespace dynd
