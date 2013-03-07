//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/memblock/memory_block.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/memblock/zeroinit_memory_block.hpp>
#include <dynd/memblock/fixed_size_pod_memory_block.hpp>
#include <dynd/memblock/executable_memory_block.hpp>
#include <dynd/memblock/ndobject_memory_block.hpp>
#include <dynd/memblock/external_memory_block.hpp>

using namespace std;
using namespace dynd;

namespace dynd { namespace detail {

/**
 * INTERNAL: Frees a memory_block created by make_external_memory_block.
 * This should only be called by the memory_block decref code.
 */
void free_external_memory_block(memory_block_data *memblock);
/**
 * INTERNAL: Frees a memory_block created by make_fixed_size_pod_memory_block.
 * This should only be called by the memory_block decref code.
 */
void free_fixed_size_pod_memory_block(memory_block_data *memblock);
/**
 * INTERNAL: Frees a memory_block created by make_pod_memory_block.
 * This should only be called by the memory_block decref code.
 */
void free_pod_memory_block(memory_block_data *memblock);
/**
 * INTERNAL: Frees a memory_block created by make_zeroinit_memory_block.
 * This should only be called by the memory_block decref code.
 */
void free_zeroinit_memory_block(memory_block_data *memblock);
/**
 * INTERNAL: Frees a memory_block created by make_executable_memory_block.
 * This should only be called by the memory_block decref code.
 */
void free_executable_memory_block(memory_block_data *memblock);
/**
 * INTERNAL: Frees a memory_block created by make_ndobject_memory_block.
 * This should only be called by the memory_block decref code.
 */
void free_ndobject_memory_block(memory_block_data *memblock);
/**
 * INTERNAL: Frees a memory_block created by make_objectarray_memory_block.
 * This should only be called by the memory_block decref code.
 */
void free_objectarray_memory_block(memory_block_data *memblock);


/**
 * INTERNAL: Static instance of the pod allocator API for the POD memory block.
 */
extern memory_block_pod_allocator_api pod_memory_block_allocator_api;
/**
 * INTERNAL: Static instance of the pod allocator API for the zeroinit memory block.
 */
extern memory_block_pod_allocator_api zeroinit_memory_block_allocator_api;
/**
 * INTERNAL: Static instance of the objectarray allocator API for the objectarray memory block.
 */
extern memory_block_objectarray_allocator_api objectarray_memory_block_allocator_api;

}} // namespace dynd::detail


void dynd::detail::memory_block_free(memory_block_data *memblock)
{
    //cout << "freeing memory block " << (void *)memblock << endl;
    switch ((memory_block_type_t)memblock->m_type) {
        case external_memory_block_type: {
            free_external_memory_block(memblock);
            return;
        }
        case fixed_size_pod_memory_block_type: {
            free_fixed_size_pod_memory_block(memblock);
            return;
        }
        case pod_memory_block_type: {
            free_pod_memory_block(memblock);
            return;
        }
        case zeroinit_memory_block_type: {
            free_zeroinit_memory_block(memblock);
            return;
        }
        case objectarray_memory_block_type: {
            free_objectarray_memory_block(memblock);
            return;
        }
        case executable_memory_block_type:
            free_executable_memory_block(memblock);
            return;
        case ndobject_memory_block_type:
            free_ndobject_memory_block(memblock);
            return;
    }

    stringstream ss;
    ss << "unrecognized memory block type, " << memblock->m_type << ", likely memory corruption";
    throw runtime_error(ss.str());
}

void dynd::memory_block_debug_print(const memory_block_data *memblock, std::ostream& o, const std::string& indent)
{
    if (memblock != NULL) {
        o << indent << "------ memory_block at " << (const void *)memblock << "\n";
        o << indent << " reference count: " << (int32_t)memblock->m_use_count << "\n";
        switch ((memory_block_type_t)memblock->m_type) {
            case external_memory_block_type:
                o << indent << " type: external\n";
                external_memory_block_debug_print(memblock, o, indent);
                break;
            case fixed_size_pod_memory_block_type:
                o << indent << " type: fixed_size_pod\n";
                fixed_size_pod_memory_block_debug_print(memblock, o, indent);
                break;
            case pod_memory_block_type:
                o << indent << " type: pod\n";
                pod_memory_block_debug_print(memblock, o, indent);
                break;
            case zeroinit_memory_block_type:
                o << indent << " type: zeroinit\n";
                zeroinit_memory_block_debug_print(memblock, o, indent);
                break;
            case objectarray_memory_block_type:
                o << indent << " type: object\n";
                break;
            case executable_memory_block_type:
                o << indent << " type: executable\n";
                executable_memory_block_debug_print(memblock, o, indent);
                break;
            case ndobject_memory_block_type:
                o << indent << " type: ndobject\n";
                ndobject_memory_block_debug_print(memblock, o, indent);
                break;
        }
        o << indent << "------" << endl;
    } else {
        o << indent << "------ NULL memory block" << endl;
    }
}

memory_block_pod_allocator_api *dynd::get_memory_block_pod_allocator_api(memory_block_data *memblock)
{
    switch (memblock->m_type) {
        case external_memory_block_type:
            throw runtime_error("Cannot get a POD allocator API from an external_memory_block");
        case fixed_size_pod_memory_block_type:
            throw runtime_error("Cannot get a POD allocator API from an fixed_size_pod_memory_block");
        case pod_memory_block_type:
            return &dynd::detail::pod_memory_block_allocator_api;
        case zeroinit_memory_block_type:
            return &dynd::detail::zeroinit_memory_block_allocator_api;
        case objectarray_memory_block_type:
            throw runtime_error("Cannot get a POD allocator API from an objectarray_memory_block");
        case executable_memory_block_type:
            throw runtime_error("Cannot get a POD allocator API from an executable_memory_block");
        default:
            throw runtime_error("unknown memory block type");
    }
}

memory_block_objectarray_allocator_api *dynd::get_memory_block_objectarray_allocator_api(
                memory_block_data *memblock)
{
    switch (memblock->m_type) {
        case external_memory_block_type:
            throw runtime_error("Cannot get an objectarray allocator API from an external_memory_block");
        case fixed_size_pod_memory_block_type:
            throw runtime_error("Cannot get an objectarray allocator API from an fixed_size_pod_memory_block");
        case pod_memory_block_type:
            throw runtime_error("Cannot get an objectarray allocator API from a pod_memory_block");
        case zeroinit_memory_block_type:
            throw runtime_error("Cannot get an objectarray allocator API from a zeroinit_memory_block");
        case objectarray_memory_block_type:
            return &dynd::detail::objectarray_memory_block_allocator_api;
        case executable_memory_block_type:
            throw runtime_error("Cannot get an objectarray allocator API from an executable_memory_block");
        default:
            throw runtime_error("unknown memory block type");
    }
}