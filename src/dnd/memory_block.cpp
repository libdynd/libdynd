//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/memory_block.hpp>
#include <dnd/nodes/ndarray_node.hpp>

using namespace std;
using namespace dnd;

namespace {
    struct external_memory_block {
        /** Every memory block object needs this at the front */
        memory_block_data m_mbd;
        /** A void pointer for the external object */
        void *m_object;
        /** A function which frees the external object */
        external_memory_block_free_t m_free_fn;

        explicit external_memory_block(long use_count, memory_block_type_t type, void *object, external_memory_block_free_t free_fn)
            : m_mbd(use_count, type), m_object(object), m_free_fn(free_fn)
        {
        }
    };
} // anonymous namespace

void dnd::detail::memory_block_free(memory_block_data *memblock)
{
    switch ((memory_block_type_t)memblock->m_type) {
        case ndarray_node_memory_block_type:
            throw runtime_error("ndarray_node_memory_block_type not supported yet");
        case external_memory_block_type: {
            external_memory_block *emb = reinterpret_cast<external_memory_block *>(memblock);
            emb->m_free_fn(emb->m_object);
            delete emb;
            return;
        }
        case fixed_size_pod_memory_block_type: {
            free(reinterpret_cast<void *>(memblock));
            return;
        }
        case pod_memory_block_type:
            throw runtime_error("pod_memory_block_type not supported yet");
        case object_memory_block_type:
            throw runtime_error("object_memory_block_type not supported yet");
    }

    stringstream ss;
    ss << "unrecognized memory block type, " << memblock->m_type << ", likely memory corruption";
    throw runtime_error(ss.str());
}

void dnd::memory_block_debug_dump(const memory_block_data *memblock, std::ostream& o, const std::string& indent)
{
    o << indent << "------ memory_block at " << (const void *)memblock << "\n";
    o << indent << " reference count: " << memblock->m_use_count << "\n";
    switch ((memory_block_type_t)memblock->m_type) {
        case ndarray_node_memory_block_type:
            o << indent << " type: ndarray_node\n";
            break;
        case external_memory_block_type:
            o << indent << " type: external\n";
            break;
        case fixed_size_pod_memory_block_type:
            o << indent << " type: fixed_size_pod\n";
            break;
        case pod_memory_block_type:
            o << indent << " type: pod\n";
            break;
        case object_memory_block_type:
            o << indent << " type: object\n";
            break;
    }
    o << indent << "------" << endl;
}

memory_block_ref dnd::make_external_memory_block(void *object, external_memory_block_free_t free_fn)
{
    external_memory_block *emb = new external_memory_block(1, external_memory_block_type, object, free_fn);
    return memory_block_ref(reinterpret_cast<memory_block_data *>(emb), false);
}

memory_block_ref dnd::make_fixed_size_pod_memory_block(intptr_t alignment, intptr_t size, char **out_datapointer)
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
    return memory_block_ref(new (result) memory_block_data(1, fixed_size_pod_memory_block_type), false);
}