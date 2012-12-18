//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/memblock/fixed_size_pod_memory_block.hpp>

using namespace std;
using namespace dynd;

namespace dynd { namespace detail {

void free_fixed_size_pod_memory_block(memory_block_data *memblock)
{
    free(reinterpret_cast<void *>(memblock));
}

}} // namespace dynd::detail

memory_block_ptr dynd::make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment, char **out_datapointer)
{
    // Calculate the aligned starting point for the data
    intptr_t start = (intptr_t)(((uintptr_t)sizeof(memory_block_data) + (uintptr_t)(alignment - 1))
                        & ~((uintptr_t)(alignment - 1)));
    // Allocate it
    char *result = (char *)malloc(start + size_bytes);
    if (result == 0) {
        throw bad_alloc();
    }
    // Give back the data pointer
    *out_datapointer = result + start;
    // Use placement new to initialize and return the memory block
    return memory_block_ptr(new (result) memory_block_data(1, fixed_size_pod_memory_block_type), false);
}

void dynd::fixed_size_pod_memory_block_debug_print(const memory_block_data *memblock, std::ostream& o, const std::string& indent)
{
}
