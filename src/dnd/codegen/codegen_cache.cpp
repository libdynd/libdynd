//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/codegen/codegen_cache.hpp>
#include <dnd/memblock/executable_memory_block.hpp>

using namespace std;
using namespace dnd;

dnd::codegen_cache::codegen_cache()
    : m_exec_memblock(make_executable_memory_block())
{
}
