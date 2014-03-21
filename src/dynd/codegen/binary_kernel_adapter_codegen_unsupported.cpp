//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/platform_definitions.hpp>

#if !defined(DYND_CALL_MSFT_X64) && !defined(DYND_CALL_MSFT_X32) && !defined(DYND_CALL_SYSV_X64)
#include <dynd/codegen/binary_kernel_adapter_codegen.hpp>

#include <stdexcept>

#if 0

namespace dynd
{

namespace
{
    void unimplemented()
    {
        throw std::runtime_error("unimplemented error");
    }
}
uint64_t
get_binary_function_adapter_unique_id( const ndt::type& DYND_UNUSED(restype)
                                      , const ndt::type& DYND_UNUSED(arg0type)
                                      , const ndt::type& DYND_UNUSED(arg1type)
                                      , calling_convention_t DYND_UNUSED(callconv))
{
    unimplemented();
    return 0;
}

std::string
get_binary_function_adapter_unique_id_string(uint64_t DYND_UNUSED(unique_id))
{
    unimplemented();
    return std::string();
}
    
binary_operation_pair_t codegen_binary_function_adapter(
                const memory_block_ptr& DYND_UNUSED(exec_memblock), const ndt::type& DYND_UNUSED(restype),
                const ndt::type& DYND_UNUSED(arg0type), const ndt::type& DYND_UNUSED(arg1type), calling_convention_t DYND_UNUSED(callconv))
{
    unimplemented();
    return binary_operation_pair_t();
}

}
#endif

#endif

