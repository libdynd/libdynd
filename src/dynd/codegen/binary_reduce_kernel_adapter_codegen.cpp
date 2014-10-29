//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#if 0 // Temporarily disabled

#include <dynd/codegen/binary_reduce_kernel_adapter_codegen.hpp>

using namespace std;
using namespace dynd;

static unsigned int get_arg_id_from_type_id(type_id_t type_id)
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

uint64_t dynd::get_binary_reduce_function_adapter_unique_id(const ndt::type& reduce_type, calling_convention_t DYND_UNUSED(callconv))
{
    uint64_t result = get_arg_id_from_type_id(reduce_type.get_type_id());

#if defined(_WIN32) && !defined(_M_X64)
    // For 32-bit Windows, support both cdecl and stdcall
    result += (uint64_t)callconv << 4;
#endif

    return result;
}

std::string dynd::get_binary_reduce_function_adapter_unique_id_string(uint64_t unique_id)
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
    template<class T>
    struct binary_reduce_function_adapters {
        typedef T (*cdecl_func_ptr_t)(T, T);

        static void single_left_associative(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            cdecl_func_ptr_t kfunc = get_auxiliary_data<cdecl_func_ptr_t>(extra->auxdata);
            *reinterpret_cast<T *>(dst) = kfunc(*reinterpret_cast<T *>(dst),
                                            *reinterpret_cast<const T *>(src));
        }

        static void strided_left_associative(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                        size_t count, unary_kernel_static_data *extra)
        {
            cdecl_func_ptr_t kfunc = get_auxiliary_data<cdecl_func_ptr_t>(extra->auxdata);
            for (size_t i = 0; i != count; ++i) {
                *reinterpret_cast<T *>(dst) = kfunc(*reinterpret_cast<T *>(dst),
                                                *reinterpret_cast<const T *>(src));
                dst += dst_stride;
                src += src_stride;
            }
        }

        static void single_right_associative(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            cdecl_func_ptr_t kfunc = get_auxiliary_data<cdecl_func_ptr_t>(extra->auxdata);
            *reinterpret_cast<T *>(dst) = kfunc(*reinterpret_cast<const T *>(src),
                                            *reinterpret_cast<T *>(dst));
        }

        static void strided_right_associative(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                        size_t count, unary_kernel_static_data *extra)
        {
            cdecl_func_ptr_t kfunc = get_auxiliary_data<cdecl_func_ptr_t>(extra->auxdata);
            for (size_t i = 0; i != count; ++i) {
                *reinterpret_cast<T *>(dst) = kfunc(*reinterpret_cast<const T *>(src),
                                                *reinterpret_cast<T *>(dst));
                dst += dst_stride;
                src += src_stride;
            }
        }
    };
} // anonymous namespace

unary_operation_pair_t dynd::codegen_left_associative_binary_reduce_function_adapter(
                    const ndt::type& reduce_type, calling_convention_t DYND_UNUSED(callconv))
{
    // TODO: If there's a platform where there are differences in the calling convention
    //       between the equated types, this will have to change.
    switch(reduce_type.get_type_id()) {
        case bool_type_id:
        case int8_type_id:
        case uint8_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<int8_t>::single_left_associative,
                            &binary_reduce_function_adapters<int8_t>::strided_left_associative);
        case int16_type_id:
        case uint16_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<int16_t>::single_left_associative,
                            &binary_reduce_function_adapters<int16_t>::strided_left_associative);
        case int32_type_id:
        case uint32_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<int32_t>::single_left_associative,
                            &binary_reduce_function_adapters<int32_t>::strided_left_associative);
        case int64_type_id:
        case uint64_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<int64_t>::single_left_associative,
                            &binary_reduce_function_adapters<int64_t>::strided_left_associative);
        case float32_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<float>::single_left_associative,
                            &binary_reduce_function_adapters<float>::strided_left_associative);
        case float64_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<double>::single_left_associative,
                            &binary_reduce_function_adapters<double>::strided_left_associative);
        default: {
            stringstream ss;
            ss << "The binary reduce function adapter does not support " << reduce_type;
            throw runtime_error(ss.str());
        }
    }
}

unary_operation_pair_t dynd::codegen_right_associative_binary_reduce_function_adapter(
                    const ndt::type& reduce_type, calling_convention_t DYND_UNUSED(callconv))
{
    // TODO: If there's a platform where there are differences in the calling convention
    //       between the equated types, this will have to change.
    switch(reduce_type.get_type_id()) {
        case bool_type_id:
        case int8_type_id:
        case uint8_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<int8_t>::single_right_associative,
                            &binary_reduce_function_adapters<int8_t>::strided_right_associative);
        case int16_type_id:
        case uint16_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<int16_t>::single_right_associative,
                            &binary_reduce_function_adapters<int16_t>::strided_right_associative);
        case int32_type_id:
        case uint32_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<int32_t>::single_right_associative,
                            &binary_reduce_function_adapters<int32_t>::strided_right_associative);
        case int64_type_id:
        case uint64_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<int64_t>::single_right_associative,
                            &binary_reduce_function_adapters<int64_t>::strided_right_associative);
        case float32_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<float>::single_right_associative,
                            &binary_reduce_function_adapters<float>::strided_right_associative);
        case float64_type_id:
            return unary_operation_pair_t(&binary_reduce_function_adapters<double>::single_right_associative,
                            &binary_reduce_function_adapters<double>::strided_right_associative);
        default: {
            stringstream ss;
            ss << "The binary reduce function adapter does not support " << reduce_type;
            throw runtime_error(ss.str());
        }
    }
}

#endif // temporarily disabled

