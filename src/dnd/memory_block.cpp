//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/memory_block.hpp>
#include <dnd/nodes/ndarray_expr_node.hpp>

using namespace std;
using namespace dnd;

void dnd::detail::memory_block_free(memory_block_data *memblock)
{
    switch ((memory_block_type_t)memblock->m_type) {
        case ndarray_node_memory_block_type:
            throw runtime_error("ndarray_node_memory_block_type not supported yet");
        case external_memory_block_type:
            throw runtime_error("external_memory_block_type not supported yet");
        case fixed_size_pod_memory_block_type:
            throw runtime_error("fixed_size_pod_memory_block_type not supported yet");
        case pod_memory_block_type:
            throw runtime_error("pod_memory_block_type not supported yet");
        case object_memory_block_type:
            throw runtime_error("object_memory_block_type not supported yet");
    }

    stringstream ss;
    ss << "unrecognized memory block type, " << memblock->m_type << ", likely memory corruption";
    throw runtime_error(ss.str());
}
