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
   * Creates a memory block which can be used to allocate zero-initialized
   * object type output memory for blockref types.
   *
   * The initial count of elements can be set if a good estimate is known.
   *
   * \param dt  The data type of the objects to allocate.
   * \param arrmeta  The arrmeta corresponding to the data type for the objects to allocate.
   * \param stride  For objects without a fixed size, the size of the memory to allocate
   *                for each element. This would be typically set to the value for
   *                get_default_data_size() corresponding to default-constructed arrmeta.
   * \param initial_count  The number of elements to allocate at the start.
   */
  DYNDT_API memory_block make_objectarray_memory_block(const ndt::type &dt, const char *arrmeta, intptr_t stride,
                                                       intptr_t initial_count = 64, size_t arrmeta_size = 0);

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
