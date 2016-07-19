//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/memblock/base_memory_block.hpp>

namespace dynd {
namespace nd {

  class DYNDT_API memory_block : public intrusive_ptr<base_memory_block> {
  public:
    using intrusive_ptr<base_memory_block>::intrusive_ptr;

    memory_block() = default;

    memory_block(const buffer &other);
  };

  template <typename T, typename... ArgTypes>
  memory_block make_memory_block(ArgTypes &&... args) {
    return memory_block(new T(std::forward<ArgTypes>(args)...), false);
  }

} // namespace dynd::nd
} // namespace dynd
