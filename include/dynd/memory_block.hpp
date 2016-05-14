//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/buffer.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {
namespace nd {

  class DYNDT_API memory_block : public intrusive_ptr<memory_block_data> {
  public:
    using intrusive_ptr<memory_block_data>::intrusive_ptr;

    memory_block() = default;

    memory_block(const buffer &other);
  };

  template <typename T, typename... ArgTypes>
  memory_block make_memory_block(ArgTypes &&... args) {
    return memory_block(new T(std::forward<ArgTypes>(args)...), false);
  }

  /**
   * Creates a memory block of a pre-determined fixed size. A pointer to the
   * memory allocated for data is placed in the output parameter.
   */
  DYNDT_API intrusive_ptr<memory_block_data> make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment,
                                                                              char **out_datapointer);

  /**
   * Creates a memory block which can be used to allocate zero-initialized
   * POD output memory for blockref types.
   *
   * The initial capacity can be set if a good estimate is known.
   */
  DYNDT_API memory_block make_zeroinit_memory_block(const ndt::type &element_tp,
                                                    intptr_t initial_capacity_bytes = 2048);

} // namespace dynd::nd
} // namespace dynd
