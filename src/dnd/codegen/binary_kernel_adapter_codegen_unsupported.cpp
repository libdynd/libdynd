//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/platform_definitions.h>

#if !defined(DND_CALL_MSFT_X64) && !defined(DND_CALL_SYSV_X64)
#include <dnd/codegen/binary_kernel_adapter_codegen.hpp>

#include <stdexcept>

namespace dnd
{

namespace
{
    void unimplemented()
    {
        throw std::runtime_error("unimplemented error");
    }
}
uint64_t
get_binary_function_adapter_unique_id( const dtype& DND_UNUSED(restype)
                                      , const dtype& DND_UNUSED(arg0type)
                                      , const dtype& DND_UNUSED(arg1type)
                                      , calling_convention_t DND_UNUSED(callconv))
{
    unimplemented();
    return 0;
}

std::string
get_binary_function_adapter_unique_id_string(uint64_t DND_UNUSED(unique_id))
{
    unimplemented();
    return std::string();
}
    
binary_operation_t
codegen_binary_function_adapter(const memory_block_ptr& DND_UNUSED(exec_memblock)
                                , const dtype& DND_UNUSED(restype)
                                , const dtype& DND_UNUSED(arg0type)
                                , const dtype& DND_UNUSED(arg1type)
                                , calling_convention_t DND_UNUSED(callconv))
{
    unimplemented();
    return 0;
}

}

#endif

