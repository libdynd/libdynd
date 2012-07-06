//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/memblock/external_memory_block.hpp>

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

memory_block_ptr dnd::make_external_memory_block(void *object, external_memory_block_free_t free_fn)
{
    external_memory_block *emb = new external_memory_block(1, external_memory_block_type, object, free_fn);
    return memory_block_ptr(reinterpret_cast<memory_block_data *>(emb), false);
}

void free_external_memory_block(memory_block_data *memblock)
{
    external_memory_block *emb = reinterpret_cast<external_memory_block *>(memblock);
    emb->m_free_fn(emb->m_object);
    delete emb;
}