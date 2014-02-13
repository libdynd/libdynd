//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BINARY_KERNEL_ADAPTER_CODEGEN_HPP_
#define _DYND__BINARY_KERNEL_ADAPTER_CODEGEN_HPP_

#include <dynd/type.hpp>
#include <dynd/memblock/memory_block.hpp>
#include <dynd/codegen/calling_conventions.hpp>

namespace dynd {

/**
 * This is the auxiliary data used by the code generated
 * binary function adapters.
 *
 * Use make_auxiliary_data<binary_function_adapter_auxdata>(out_auxdata)
 * to create it, then populate with the function pointer and the memory
 * block containing the code.
 */
struct binary_function_adapter_auxdata {
    void *function_pointer;
    memory_block_ptr adapter_memblock, adaptee_memblock;
};

/**
 * This returns an integer ID that uniquely identifies the
 * function adapter produced by codegen_binary_function_adapter.
 * If two sets of inputs produce the same unique ID, they would also
 * produce the same generated code.
 */
uint64_t get_binary_function_adapter_unique_id(const ndt::type& restype,
                    const ndt::type& arg0type, const ndt::type& arg1type, calling_convention_t callconv);

/**
 * Gets the unique integer ID in a string form, hopefully in human
 * readable form.
 */
std::string get_binary_function_adapter_unique_id_string(uint64_t unique_id);

/**
 * Gets a kernel for adapting a binary function pointer of the given
 * prototype.
 *
 * @param exec_memblock  An executable_memory_block where memory for the
 *                       code generation is used.
 * @param restype        The return type of the function.
 * @param arg0type       The type of the function's first parameter.
 * @param arg1type       The type of the function's second parameter.
 * @param callconv       The calling convention of the function to adapt.
 *
 * @return A pointer to the binary adapter kernel.
 */
//binary_operation_pair_t codegen_binary_function_adapter(const memory_block_ptr& exec_memblock, const ndt::type& restype,
//                    const ndt::type& arg0type, const ndt::type& arg1type, calling_convention_t callconv);

} // namespace dynd

#endif // _DYND__BINARY_KERNEL_ADAPTER_CODEGEN_HPP_
