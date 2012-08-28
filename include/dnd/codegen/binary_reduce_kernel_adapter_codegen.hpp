//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__BINARY_REDUCE_KERNEL_ADAPTER_CODEGEN_HPP_
#define _DND__BINARY_REDUCE_KERNEL_ADAPTER_CODEGEN_HPP_

#include <dnd/dtype.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/memblock/memory_block.hpp>
#include <dnd/codegen/calling_conventions.hpp>

namespace dnd {

/**
 * This is the auxiliary data used by the code generated
 * binary function adapters.
 *
 * TODO: They're all the same, move them to a common header and
 *       make them actually be just one struct.
 *
 * Use make_auxiliary_data<binary_function_adapter_auxdata>(out_auxdata)
 * to create it, then populate with the function pointer and the memory
 * block containing the code.
 *
 * @param function_pointer  Pointer to the function being adapted.
 * @param adapter_memblock  Empty, or a reference to the object holding the adapter code memory.
 * @param adaptee_memblock  Empty, or a reference to the object holding the function being adapted.
 */
struct binary_reduce_function_adapter_auxdata {
    void *function_pointer;
    memory_block_ptr adapter_memblock, adaptee_memblock;
};

/**
 * This returns an integer ID that uniquely identifies the
 * function adapter produced by codegen_unary_function_adapter.
 * If two sets of inputs produce the same unique ID, they would also
 * produce the same generated code.
 */
uint64_t get_binary_reduce_function_adapter_unique_id(const dtype& reduce_type, calling_convention_t callconv);

/**
 * Gets the unique integer ID in a string form, hopefully in human
 * readable form.
 */
std::string get_binary_reduce_function_adapter_unique_id_string(uint64_t unique_id);

/**
 * Gets a kernel for adapting a binary function pointer of the given
 * prototype as a unary reduction kernel, assuming a left associative
 * reduction iterating from indices 0 to N-1.
 *
 * @param exec_memblock  An executable_memory_block where memory for the
 *                       code generation is used.
 * @param reduce_type    The return type and the type of the two parameters.
 * @param callconv       The calling convention of the function to adapt.
 *
 * @return A pointer to the binary adapter kernel.
 */
unary_operation_t codegen_left_associative_binary_reduce_function_adapter(
                    const dtype& reduce_type,calling_convention_t callconv);

/**
 * Gets a kernel for adapting a binary function pointer of the given
 * prototype as a unary reduction kernel, assuming a right associative
 * reduction iterating from indices N-1 to 0.
 *
 * @param exec_memblock  An executable_memory_block where memory for the
 *                       code generation is used.
 * @param reduce_type    The return type and the type of the two parameters.
 * @param callconv       The calling convention of the function to adapt.
 *
 * @return A pointer to the binary adapter kernel.
 */
unary_operation_t codegen_right_associative_binary_reduce_function_adapter(
                    const dtype& reduce_type,calling_convention_t callconv);

} // namespace dnd

#endif // _DND__BINARY_REDUCE_KERNEL_ADAPTER_CODEGEN_HPP_
