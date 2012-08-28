//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__UNARY_KERNEL_ADAPTER_CODEGEN_HPP_
#define _DND__UNARY_KERNEL_ADAPTER_CODEGEN_HPP_

#include <dnd/dtype.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/memblock/memory_block.hpp>
#include <dnd/codegen/calling_conventions.hpp>

namespace dnd {

/**
 * This is the auxiliary data used by the code generated
 * unary function adapters.
 *
 * Use make_auxiliary_data<unary_function_adapter_auxdata>(out_auxdata)
 * to create it, then populate with the function pointer and the memory
 * block containing the code.
 */
struct unary_function_adapter_auxdata {
    void *function_pointer;
    memory_block_ptr adapter_memblock, adaptee_memblock;
};

/**
 * This returns an integer ID that uniquely identifies the
 * unary function adapter produced by codegen_unary_function_adapter.
 * If two sets of inputs produce the same unique ID, they would also
 * produce the same generated code.
 */
uint64_t get_unary_function_adapter_unique_id(const dtype& restype,
                    const dtype& arg0type, calling_convention_t callconv);

/**
 * Gets the unique integer ID in a string form, hopefully in human
 * readable form.
 */
std::string get_unary_function_adapter_unique_id_string(uint64_t unique_id);

/**
 * Gets a kernel for adapting a unary function pointer of the given
 * prototype.
 *
 * @param exec_memblock  An executable_memory_block where memory for the
 *                       code generation is used.
 * @param restype        The return type of the function.
 * @param arg0type       The type of the function's first parameter.
 * @param callconv       The calling convention of the function to adapt.
 *
 * @return An array of four unary_operation_t pointers, matching to the unary_specialization_t enum.
 *         These may all be the same pointer if no specialization was done.
 */
unary_operation_t* codegen_unary_function_adapter(const memory_block_ptr& exec_memblock, const dtype& restype,
                    const dtype& arg0type, calling_convention_t callconv);

} // namespace dnd

#endif // _DND__UNARY_KERNEL_ADAPTER_CODEGEN_HPP_
