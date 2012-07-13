//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__CODEGEN_CACHE_HPP_
#define _DND__CODEGEN_CACHE_HPP_

#include <dnd/codegen/unary_kernel_adapter_codegen.hpp>

namespace dnd {

/**
 * This class owns an executable_memory_block, and provides a caching
 * interface to kernel adapters that require codegen.
 */
class codegen_cache {
    memory_block_ptr m_exec_memblock;

public:
    codegen_cache();
};

} // namespace dnd

#endif // _DND__CODEGEN_CACHE_HPP_
