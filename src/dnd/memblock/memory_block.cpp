//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/memblock/memory_block.hpp>
#include <dnd/memblock/pod_memory_block.hpp>
#include <dnd/memblock/fixed_size_pod_memory_block.hpp>
#include <dnd/nodes/ndarray_node.hpp>

using namespace std;
using namespace dnd;

namespace dnd { namespace detail {

/**
 * INTERNAL: Frees a memory_block created by make_ndarray_node_memory_block.
 * This should only be called by the memory_block decref code.
 */
void free_ndarray_node_memory_block(memory_block_data *memblock);
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
 * INTERNAL: Static instance of the pod allocator API for the POD memory block.
 */
extern memory_block_pod_allocator_api pod_memory_block_allocator_api;

}} // namespace dnd::detail


void dnd::detail::memory_block_free(memory_block_data *memblock)
{
    //cout << "freeing memory block " << (void *)memblock << endl;
    switch ((memory_block_type_t)memblock->m_type) {
        case ndarray_node_memory_block_type: {
            free_ndarray_node_memory_block(memblock);
            return;
        }
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
        case object_memory_block_type:
            throw runtime_error("object_memory_block_type not supported yet");
    }

    stringstream ss;
    ss << "unrecognized memory block type, " << memblock->m_type << ", likely memory corruption";
    throw runtime_error(ss.str());
}

void dnd::memory_block_debug_dump(const memory_block_data *memblock, std::ostream& o, const std::string& indent)
{
    if (memblock != NULL) {
        o << indent << "------ memory_block at " << (const void *)memblock << "\n";
        o << indent << " reference count: " << memblock->m_use_count << "\n";
        switch ((memory_block_type_t)memblock->m_type) {
            case ndarray_node_memory_block_type: {
                o << indent << " type: ndarray_node\n";
                ndarray_node_ptr node(const_cast<memory_block_data *>(memblock));
                node->debug_dump(o, indent + " ");
                break;
            }
            case external_memory_block_type:
                o << indent << " type: external\n";
                break;
            case fixed_size_pod_memory_block_type:
                o << indent << " type: fixed_size_pod\n";
                fixed_size_pod_memory_block_debug_dump(memblock, o, indent);
                break;
            case pod_memory_block_type:
                o << indent << " type: pod\n";
                pod_memory_block_debug_dump(memblock, o, indent);
                break;
            case object_memory_block_type:
                o << indent << " type: object\n";
                break;
        }
        o << indent << "------" << endl;
    } else {
        o << indent << "------ NULL memory block" << endl;
    }
}

memory_block_pod_allocator_api *dnd::get_memory_block_pod_allocator_api(memory_block_data *memblock)
{
    switch (memblock->m_type) {
        case ndarray_node_memory_block_type:
            throw runtime_error("Cannot get a POD allocator API from an ndarray_node_memory_block");
        case external_memory_block_type:
            throw runtime_error("Cannot get a POD allocator API from an external_memory_block");
        case fixed_size_pod_memory_block_type:
            throw runtime_error("Cannot get a POD allocator API from an fixed_size_pod_memory_block");
        case pod_memory_block_type:
            return &dnd::detail::pod_memory_block_allocator_api;
        case object_memory_block_type:
            throw runtime_error("Cannot get a POD allocator API from an object_memory_block");
        default:
            throw runtime_error("unknown memory block type");
    }
}
