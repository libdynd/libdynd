//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__CODEGEN_CACHE_HPP_
#define _DND__CODEGEN_CACHE_HPP_

#include <map>

#include <dnd/dtype.hpp>
#include <dnd/kernels/unary_kernel_instance.hpp>
#include <dnd/codegen/calling_conventions.hpp>

namespace dnd {

/**
 * This class owns an executable_memory_block, and provides a caching
 * interface to kernel adapters that require codegen.
 */
class codegen_cache {
    /** The memory block all the generated code goes into */
    memory_block_ptr m_exec_memblock;
    /** A mapping from unary kernel adapter unique id to the generated kernel adapter */
    std::map<uint64_t, unary_operation_t *> m_cached_unary_kernel_adapters;
public:
    codegen_cache();

    /**
     * Returns the executable memory block that
     * this codegen cache generates into.
     */
    const memory_block_ptr& get_exec_memblock()
    {
        return m_exec_memblock;
    }

    /**
     * Generates the requested unary function adapter, and returns a
     * pointer to a specialized_unary_operation_table_t embedded in the
     * codegen_cache.
     */
    unary_operation_t* codegen_unary_function_adapter(const dtype& restype,
                    const dtype& arg0type, calling_convention_t callconv);
};

} // namespace dnd

#endif // _DND__CODEGEN_CACHE_HPP_
