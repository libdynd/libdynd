//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

// DEPRECATED 2012-11-02

#ifndef _DYND__NDARRAY_NODE_MEMORY_BLOCK_HPP_
#define _DYND__NDARRAY_NODE_MEMORY_BLOCK_HPP_

#include <dynd/memblock/memory_block.hpp>
#include <dynd/nodes/ndarray_node.hpp>

namespace dynd {

/**
 * Creates a memory block for an ndarray_node. The caller must call
 * placement new on the embedded ndarray_node subclass before using it.
 */
ndarray_node_ptr make_uninitialized_ndarray_node_memory_block(intptr_t sizeof_node, char **out_node_memory);


} // namespace dynd

#endif // _DYND__NDARRAY_NODE_MEMORY_BLOCK_HPP_
