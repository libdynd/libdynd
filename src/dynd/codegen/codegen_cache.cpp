//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/codegen/codegen_cache.hpp>
#include <dynd/memblock/executable_memory_block.hpp>
#include <dynd/codegen/unary_kernel_adapter_codegen.hpp>
#include <dynd/codegen/binary_kernel_adapter_codegen.hpp>
#include <dynd/codegen/binary_reduce_kernel_adapter_codegen.hpp>
#include <dynd/kernels/unary_kernel_instance.hpp>

using namespace std;
using namespace dynd;

dynd::codegen_cache::codegen_cache()
    : m_exec_memblock(make_executable_memory_block()),
        m_cached_unary_kernel_adapters(),
        m_cached_binary_kernel_adapters()
{
}

void dynd::codegen_cache::codegen_unary_function_adapter(const dtype& restype,
                const dtype& arg0type, calling_convention_t callconv,
                void *function_pointer,
                memory_block_data *function_pointer_owner,
                unary_specialization_kernel_instance& out_kernel)
{
    // Retrieve a unary function adapter from the cache
    uint64_t unique_id = get_unary_function_adapter_unique_id(restype, arg0type, callconv);
    map<uint64_t, unary_operation_t *>::iterator it = m_cached_unary_kernel_adapters.find(unique_id);
    if (it == m_cached_unary_kernel_adapters.end()) {
        unary_operation_t *optable = ::codegen_unary_function_adapter(m_exec_memblock, restype, arg0type, callconv);
        it = m_cached_unary_kernel_adapters.insert(std::pair<uint64_t, unary_operation_t *>(unique_id, optable)).first;
    }
    // Set the specializations and create auxiliary data in the output
    out_kernel.specializations = it->second;
    make_auxiliary_data<unary_function_adapter_auxdata>(out_kernel.auxdata);
    unary_function_adapter_auxdata& ad = out_kernel.auxdata.get<unary_function_adapter_auxdata>();
    // Populate the auxiliary data
    ad.function_pointer = function_pointer;
    // Set the memblock tracking the lifetime of the generated adapter function
    ad.adapter_memblock = m_exec_memblock;
    // Set the memblock tracking the lifetime of the function being adapted
    ad.adaptee_memblock = function_pointer_owner;
}

void dynd::codegen_cache::codegen_binary_function_adapter(const dtype& restype,
                const dtype& arg0type, const dtype& arg1type,
                calling_convention_t callconv,
                void *function_pointer,
                memory_block_data *function_pointer_owner,
                kernel_instance<binary_operation_t>& out_kernel)
{
    // Retrieve a binary function adapter from the cache
    uint64_t unique_id = get_binary_function_adapter_unique_id(restype, arg0type, arg1type, callconv);
    map<uint64_t, binary_operation_t>::iterator it = m_cached_binary_kernel_adapters.find(unique_id);
    if (it == m_cached_binary_kernel_adapters.end()) {
        binary_operation_t op = ::codegen_binary_function_adapter(m_exec_memblock, restype, arg0type, arg1type, callconv);
        it = m_cached_binary_kernel_adapters.insert(std::pair<uint64_t, binary_operation_t>(unique_id, op)).first;
    }
    // Set the kernel function and create auxiliary data in the output
    out_kernel.kernel = it->second;
    make_auxiliary_data<binary_function_adapter_auxdata>(out_kernel.auxdata);
    binary_function_adapter_auxdata& ad = out_kernel.auxdata.get<binary_function_adapter_auxdata>();
    // Populate the auxiliary data
    ad.function_pointer = function_pointer;
    // Set the memblock tracking the lifetime of the generated adapter function
    ad.adapter_memblock = m_exec_memblock;
    // Set the memblock tracking the lifetime of the function being adapted
    ad.adaptee_memblock = function_pointer_owner;
}

void dynd::codegen_cache::codegen_left_associative_binary_reduce_function_adapter(
                const dtype& reduce_type,calling_convention_t callconv,
                void *function_pointer,
                memory_block_data *function_pointer_owner,
                kernel_instance<unary_operation_t>& out_kernel)
{
    // These kernels are generated with templates instead of runtime
    // codegen, so no caching is needed
    out_kernel.kernel = ::codegen_left_associative_binary_reduce_function_adapter(reduce_type, callconv);
    make_auxiliary_data<binary_reduce_function_adapter_auxdata>(out_kernel.auxdata);
    binary_reduce_function_adapter_auxdata& ad = out_kernel.auxdata.get<binary_reduce_function_adapter_auxdata>();
    // Populate the auxiliary data
    ad.function_pointer = function_pointer;
    // The adapter is static, so the adapter_memblock isn't needed
    // Set the memblock tracking the lifetime of the function being adapted
    ad.adaptee_memblock = function_pointer_owner;
}

void dynd::codegen_cache::codegen_right_associative_binary_reduce_function_adapter(
                const dtype& reduce_type,calling_convention_t callconv,
                void *function_pointer,
                memory_block_data *function_pointer_owner,
                kernel_instance<unary_operation_t>& out_kernel)
{
    // These kernels are generated with templates instead of runtime
    // codegen, so no caching is needed
    out_kernel.kernel = ::codegen_right_associative_binary_reduce_function_adapter(reduce_type, callconv);
    make_auxiliary_data<binary_reduce_function_adapter_auxdata>(out_kernel.auxdata);
    binary_reduce_function_adapter_auxdata& ad = out_kernel.auxdata.get<binary_reduce_function_adapter_auxdata>();
    // Populate the auxiliary data
    ad.function_pointer = function_pointer;
    // The adapter is static, so the adapter_memblock isn't needed
    // Set the memblock tracking the lifetime of the function being adapted
    ad.adaptee_memblock = function_pointer_owner;
}

void dynd::codegen_cache::debug_print(std::ostream& o, const std::string& indent) const
{
    o << indent << "------ codegen_cache\n";
    o << indent << " cached unary_kernel_adapters:\n";
    for (map<uint64_t, unary_operation_t *>::const_iterator i = m_cached_unary_kernel_adapters.begin(),
                i_end = m_cached_unary_kernel_adapters.end(); i != i_end; ++i) {
        o << indent << "  unique id: " << get_unary_function_adapter_unique_id_string(i->first) << "\n";
        o << indent << "  function ptrs: ";
        for (int j = 0; j < 4; ++j) {
            o << (void *)i->second[j] << " ";
        }
        o << "\n";
    }

    o << indent << " cached binary_kernel_adapters:\n";
    for (map<uint64_t, binary_operation_t>::const_iterator i = m_cached_binary_kernel_adapters.begin(),
                i_end = m_cached_binary_kernel_adapters.end(); i != i_end; ++i) {
        o << indent << "  unique id: " << get_binary_function_adapter_unique_id_string(i->first) << "\n";
        o << indent << "  function ptr: " << (void *)i->second << "\n";
    }

    o << indent << " executable memory block:\n";
    memory_block_debug_print(m_exec_memblock.get(), o, indent + " ");
    o << indent << "------" << endl;
}
