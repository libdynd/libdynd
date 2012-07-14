//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/codegen/codegen_cache.hpp>
#include <dnd/memblock/executable_memory_block.hpp>
#include <dnd/codegen/unary_kernel_adapter_codegen.hpp>
#include <dnd/kernels/unary_kernel_instance.hpp>

using namespace std;
using namespace dnd;

dnd::codegen_cache::codegen_cache()
    : m_exec_memblock(make_executable_memory_block())
{
}

unary_operation_t* dnd::codegen_cache::codegen_unary_function_adapter(const dtype& restype,
                    const dtype& arg0type, calling_convention_t callconv)
{
    uint64_t unique_id = get_unary_function_adapter_unique_id(restype, arg0type, callconv);
    map<uint64_t, unary_operation_t *>::iterator it = m_cached_unary_kernel_adapters.find(unique_id);
    if (it == m_cached_unary_kernel_adapters.end()) {
        unary_operation_t *optable = ::codegen_unary_function_adapter(m_exec_memblock, restype, arg0type, callconv);
        it = m_cached_unary_kernel_adapters.insert(std::pair<uint64_t, unary_operation_t *>(unique_id, optable)).first;
    }
    return it->second;
}
