//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/memblock/array_memory_block.hpp>

namespace dynd {
namespace nd {

  class DYNDT_API buffer : public intrusive_ptr<array_preamble> {
  public:
    using intrusive_ptr<array_preamble>::intrusive_ptr;
  };

} // namespace dynd::nd
} // namespace dynd
