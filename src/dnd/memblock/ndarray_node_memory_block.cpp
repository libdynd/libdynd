//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/memblock/ndarray_node_memory_block.hpp>

using namespace std;
using namespace dnd;

ndarray_node_ptr dnd::make_uninitialized_ndarray_node_memory_block(intptr_t sizeof_node, char **out_node_memory)
{
    //cout << "allocating ndarray node size " << sizeof_node << endl;
    //cout << "sizeof memory_block_data " << sizeof(memory_block_data) << endl;
    // Allocate the memory_block
    char *result = reinterpret_cast<char *>(malloc(sizeof(memory_block_data) + sizeof_node));
    if (result == NULL) {
        throw bad_alloc();
    }
    *out_node_memory = result + sizeof(memory_block_data);
    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}


void free_ndarray_node_memory_block(memory_block_data *memblock)
{
    ndarray_node *node = reinterpret_cast<ndarray_node *>(reinterpret_cast<char *>(memblock) + sizeof(memory_block_data));
    node->~ndarray_node();
    free(reinterpret_cast<void *>(memblock));
}
