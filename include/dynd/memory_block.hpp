//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/buffer.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {
namespace nd {

  class DYND_API memory_block : public intrusive_ptr<memory_block_data> {
  public:
    using intrusive_ptr<memory_block_data>::intrusive_ptr;

    memory_block() = default;

    memory_block(const buffer &other);
  };

} // namespace dynd::nd
} // namespace dynd
