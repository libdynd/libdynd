//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/memblock/fixed_size_pod_memory_block.hpp>

using namespace std;
using namespace dnd;

namespace dnd { namespace detail {

void free_fixed_size_pod_memory_block(memory_block_data *memblock)
{
    char *rawmem = reinterpret_cast<char *>(memblock);
    int blockrefs_size = *(int *)(rawmem + sizeof(memory_block_data));
    for (int i = 0; i < blockrefs_size; ++i) {
        memory_block_decref(*(memory_block_data **)(rawmem + sizeof(memory_block_data) + (i + 1)*sizeof(void *)));
    }
    free(reinterpret_cast<void *>(memblock));
}

}} // namespace dnd::detail

memory_block_ptr dnd::make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment, char **out_datapointer)
{
    // Calculate the aligned starting point for the data
    intptr_t start = (intptr_t)(((uintptr_t)sizeof(memory_block_data) + (uintptr_t)sizeof(int) + (uintptr_t)(alignment - 1))
                        & ~((uintptr_t)(alignment - 1)));
    // Allocate it
    char *result = (char *)malloc(start + size_bytes);
    if (result == 0) {
        throw bad_alloc();
    }
    // Indicate that there are no blockrefs
    *(int *)(result + sizeof(memory_block_data)) = 0;
    // Give back the data pointer
    *out_datapointer = result + start;
    // Use placement new to initialize and return the memory block
    return memory_block_ptr(new (result) memory_block_data(1, fixed_size_pod_memory_block_type), false);
}

memory_block_ptr dnd::make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment, char **out_datapointer,
                memory_block_ptr *blockrefs_begin, memory_block_ptr *blockrefs_end)
{
    int blockrefs_size = blockrefs_end - blockrefs_begin;
    // Calculate the aligned starting point for the data
    intptr_t start = (intptr_t)(((uintptr_t)sizeof(memory_block_data) +
                                        (uintptr_t)((blockrefs_size + 1)*sizeof(void *)) +
                                        (uintptr_t)(alignment - 1))
                        & ~((uintptr_t)(alignment - 1)));
    // Allocate it
    char *result = (char *)malloc(start + size_bytes);
    if (result == 0) {
        throw bad_alloc();
    }
    // Put in the blockrefs by moving them from the input
    *(int *)(result + sizeof(memory_block_data)) = blockrefs_size;
    for (int i = 0; i < blockrefs_size; ++i) {
        *(memory_block_data **)(result + sizeof(memory_block_data) + (i + 1)*sizeof(void *)) = blockrefs_begin[i].release();
    }
    // Give back the data pointer
    *out_datapointer = result + start;
    // Use placement new to initialize and return the memory block
    return memory_block_ptr(new (result) memory_block_data(1, fixed_size_pod_memory_block_type), false);
}

void dnd::fixed_size_pod_memory_block_debug_dump(const memory_block_data *memblock, std::ostream& o, const std::string& indent)
{
    const char *rawmem = reinterpret_cast<const char *>(memblock);
    int blockrefs_size = *(const int *)(rawmem + sizeof(memory_block_data));
    if (blockrefs_size == 0) {
        o << indent << " no blockrefs\n";
    } else {
        o << indent << " " << blockrefs_size << " blockrefs\n";
        for (int i = 0; i < blockrefs_size; ++i) {
            memory_block_debug_dump(*(memory_block_data **)(rawmem + sizeof(memory_block_data) + (i + 1)*sizeof(void *)), o, indent + " ");
        }
    }
}
