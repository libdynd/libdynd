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
        offset += node->aligned_size;
        node->destroy(node);
      }
    }

    template <typename ClosureType, typename... ArgTypes>
    void emplace_back(ArgTypes &&... args) {
      size_t offset = m_size;
      m_size += aligned_size(sizeof(call_node));
      reserve(m_size);

      new (get_at<call_node>(offset)) call_node{
          [](call_node *self, kernel_builder *kb, kernel_request_t kernreq, char *data, const char *dst_arrmeta,
             size_t nsrc, const char *const *src_arrmeta) {
            ClosureType *closure =
                reinterpret_cast<ClosureType *>(reinterpret_cast<char *>(self) + aligned_size(sizeof(call_node)));
            (*closure)(*kb, kernreq, data, dst_arrmeta, nsrc, src_arrmeta);
          },
          [](call_node *self) {
            ClosureType *closure =
                reinterpret_cast<ClosureType *>(reinterpret_cast<char *>(self) + aligned_size(sizeof(call_node)));
            closure->~ClosureType();
          },
          aligned_size(sizeof(call_node)) + aligned_size(sizeof(ClosureType))};

      storagebuf<call_node, call_graph>::emplace_back_no_init<ClosureType>(std::forward<ArgTypes>(args)...);
    }

    template <typename Arg0Type>
    void emplace_back(Arg0Type &&arg0) {
      typedef remove_reference_then_cv_t<Arg0Type> closure_type;
      this->emplace_back<closure_type, Arg0Type>(std::forward<Arg0Type>(arg0));
    }
  };

} // namespace dynd::nd
} // namespace dynd
