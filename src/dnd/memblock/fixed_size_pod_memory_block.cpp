//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/memblock/fixed_size_pod_memory_block.hpp>

using namespace std;
using namespace dnd;

void free_fixed_size_pod_memory_block(memory_block_data *memblock)
{
    free(reinterpret_cast<void *>(memblock));
}

memory_block_ptr dnd::make_fixed_size_pod_memory_block(intptr_t alignment, intptr_t size, char **out_datapointer)
{
    // Calculate the aligned starting point for the data
    intptr_t start = (intptr_t)(((uintptr_t)sizeof(memory_block_data) + (uintptr_t)(alignment - 1))
                        & ~((uintptr_t)(alignment - 1)));
    // Allocate it
    char *result = (char *)malloc(start + size);
    if (result == 0) {
        throw bad_alloc();
    }
    // Give back the data pointer
    *out_datapointer = result + start;
    // Use placement new to initialize and return the memory block
    return memory_block_ptr(new (result) memory_block_data(1, fixed_size_pod_memory_block_type), false);
}

