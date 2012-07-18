//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/codegen/binary_reduce_kernel_adapter_codegen.hpp>

using namespace std;
using namespace dnd;

static unsigned int get_arg_id_from_type_id(unsigned int type_id)
{
    switch (type_id) {
        case bool_type_id:
        case int8_type_id:
        case uint8_type_id:
            return 0;
        case int16_type_id:
        case uint16_type_id:
            return 1;
        case int32_type_id:
        case uint32_type_id:
            return 2;
        case int64_type_id:
        case uint64_type_id:
            return 3;
        case float32_type_id:
            return 4;
        case float64_type_id:
            return 5;
        default: {
            stringstream ss;
            ss << "The unary_kernel_adapter does not support " << dtype(type_id) << " for the return type";
            throw runtime_error(ss.str());
        }
    }
}

uint64_t dnd::get_binary_reduce_function_adapter_unique_id(const dtype& reduce_type, calling_convention_t callconv)
{
    uint64_t result = get_arg_id_from_type_id(reduce_type.type_id());

#if defined(_WIN32) && !defined(_M_X64)
    // For 32-bit Windows, support both cdecl and stdcall
    result += (uint64_t)callconv << 4;
#endif

    return result;
}

std::string dnd::get_binary_reduce_function_adapter_unique_id_string(uint64_t unique_id)
{
    stringstream ss;
    static const char *arg_types[8] = {"int8", "int16", "int32", "int64", "float32", "float64", "(invalid)", "(invalid)"};
    const char * reduce_type = arg_types[unique_id & 0x07];
    ss << reduce_type << " (";
    ss << reduce_type << ", ";
    ss << reduce_type << ")";
    return ss.str();
}

namespace {
    // TODO: Use templates to generate all the desired adapters. The space of possible adapters is
    //       very small because the input parameters equal the return type.
} // anonymous namespace

unary_operation_t codegen_left_associative_binary_reduce_function_adapter(const memory_block_ptr& exec_memblock,
                    const dtype& reduce_type,calling_convention_t callconv)
{
    return NULL;
}

unary_operation_t codegen_right_associative_binary_reduce_function_adapter(const memory_block_ptr& exec_memblock,
                    const dtype& reduce_type,calling_convention_t callconv)
{
    return NULL;
}
