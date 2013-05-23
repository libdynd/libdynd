//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <string>

#include <dynd/memblock/external_memory_block.hpp>

using namespace dynd;

namespace {
    struct external_memory_block {
        /** Every memory block object needs this at the front */
        memory_block_data m_mbd;
        /** A void pointer for the external object */
        void *m_object;
        /** A function which frees the external object */
        external_memory_block_free_t m_free_fn;

        external_memory_block(void *object, external_memory_block_free_t free_fn)
            : m_mbd(1, external_memory_block_type), m_object(object), m_free_fn(free_fn)
        {
        }
    };
} // anonymous namespace

memory_block_ptr dynd::make_external_memory_block(void *object, external_memory_block_free_t free_fn)
{
    external_memory_block *emb = new external_memory_block(object, free_fn);
    return memory_block_ptr(reinterpret_cast<memory_block_data *>(emb), false);
}

namespace dynd { namespace detail {

void free_external_memory_block(memory_block_data *memblock)
{
    external_memory_block *emb = reinterpret_cast<external_memory_block *>(memblock);
    emb->m_free_fn(emb->m_object);
    delete emb;
}

}} // namespace dynd::detail

void dynd::external_memory_block_debug_print(const memory_block_data *memblock, std::ostream& o, const std::string& indent)
{
    const external_memory_block *mb = reinterpret_cast<const external_memory_block *>(memblock);
    o << indent << " object void pointer: " << mb->m_object << "\n";
    o << indent << " free function: " << (const void *)mb->m_free_fn << "\n";
}
