//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#if 0 // (temporarily disabled) defined(_WIN32) && defined(_M_X64)

#include <dynd/codegen/unary_kernel_adapter_codegen.hpp>
#include <dynd/memblock/executable_memory_block.hpp>

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

uint64_t dynd::get_unary_function_adapter_unique_id(const dtype& restype,
                    const dtype& arg0type, calling_convention_t DYND_UNUSED(callconv))
{
    // Bits 0..2 for the result type
    uint64_t result = get_arg_id_from_type_id(restype.get_type_id());

    // Bits 3..5 for the arg0 type
    result += get_arg_id_from_type_id(arg0type.get_type_id()) << 3;

    // There is only one calling convention on Windows x64, so it doesn't
    // need to get encoded in the unique id.

    return result;
}

std::string dynd::get_unary_function_adapter_unique_id_string(uint64_t unique_id)
{
    stringstream ss;
    static char *arg_types[8] = {"int8", "int16", "int32", "int64", "float32", "float64", "(invalid)", "(invalid)"};
    ss << arg_types[unique_id & 0x07] << " (";
    ss << arg_types[(unique_id & (0x07 << 3)) >> 3] << ")";
    return ss.str();
}

unary_operation_pair_t dynd::codegen_unary_function_adapter(const memory_block_ptr& exec_memblock, const dtype& restype,
                    const dtype& arg0type, calling_convention_t DYND_UNUSED(callconv))
{
    // This code generation always uses the same prolog structure,
    // so the unwind_info is fixed.
    static unsigned char unwind_info[] = {
            0x01, // Version 1 (bits 0..2), All flags cleared (bits 3..7)
            0x18, // Size of prolog
            0x0a, // Count of unwind codes (10, means 20 bytes)
            0x00, // Frame register (0 means not used)
            // Unwind code: finished at offset 0x18, operation code 4 (UWOP_SAVE_NONVOL),
            // register number 6 (RSI), stored at [RSP+0x50] (8 * 0x000a)
            0x18, 0x64,
            0x0a, 0x00,
            // Unwind code: finished at offset 0x18, operation code 4 (UWOP_SAVE_NONVOL),
            // register number 5 (RBP), stored at [RSP+0x48] (8 * 0x0009)
            0x18, 0x54,
            0x09, 0x00,
            // Unwind code: finished at offset 0x18, operation code 4 (UWOP_SAVE_NONVOL),
            // register number 3 (RBX), stored at [RSP+0x40] (8 * 0x0008)
            0x18, 0x34,
            0x08, 0x00,
            // Unwind code: finished at offset 0x18, operation code 2 (UWOP_ALLOC_SMALL),
            // allocation size 0x20 = 3 * 8 + 8
            0x18, 0x32,
            // Unwind code: finished at offset 0x14, operation code 0 (UWOP_PUSH_NONVOL),
            // register number 0xD = 13 (R13)
            0x14, 0xd0,
            // Unwind code: finished at offset 0x12, operation code 0 (UWOP_PUSH_NONVOL),
            // register number 0xC = 12 (R12)
            0x12, 0xC0,
            // Unwind code: finished at offset 0x10, operation code 0 (UWOP_PUSH_NONVOL),
            // register number 7 (RDI)
            0x10, 0x70
        };
    static unsigned char prolog[] = {
            0x48, 0x89, 0x5c, 0x24, 0x08,   // mov     QWORD PTR [rsp+0x08], rbx
            0x48, 0x89, 0x6c, 0x24, 0x10,   // mov     QWORD PTR [rsp+0x10], rbp
            0x48, 0x89, 0x74, 0x24, 0x18,   // mov     QWORD PTR [rsp+0x18], rsi
            0x57,                           // push    rdi
            0x41, 0x54,                     // push    r12
            0x41, 0x55,                     // push    r13
            0x48, 0x83, 0xec, 0x20          // sub     rsp, 0x20
        };
    static unsigned char loop_setup[] = {
            0x48, 0x8b, 0x44, 0x24, 0x68,   // mov     rax, QWORD PTR auxdata$[rsp] ; AUXDATA: get arg
            0x48, 0x8b, 0x74, 0x24, 0x60,   // mov     rsi, QWORD PTR count$[rsp]
            0x4d, 0x8b, 0xe1,               // mov     r12, r9
            0x48, 0x83, 0xe0, 0xfe,         // and     rax, -2                      ; AUXDATA: Remove "borrowed" bit
            0x49, 0x8b, 0xd8,               // mov     rbx, r8
            0x4c, 0x8b, 0xea,               // mov     r13, rdx
            0x48, 0x8B, 0x68, 0x20,         // mov     rbp, QWORD PTR [rax+32]      ; AUXDATA: Get function pointer
            0x48, 0x8b, 0xf9,               // mov     rdi, rcx
            0x48, 0x85, 0xf6,               // test    rsi, rsi
            0x7e, 0x00                      // jle     SHORT skip_loop (REQUIRES FIXUP)
        };
    // loop_start:
    // Begin ARG0 CHOICE [[
    static unsigned char arg0_get_int8[] = {
            0x0f, 0xb6, 0x0b                // movzx   ecx, BYTE PTR [rbx]
        };
    static unsigned char arg0_get_int16[] = {
            0x0f, 0xb7, 0x0b                // movzx   ecx, WORD PTR [rbx]
        };
    static unsigned char arg0_get_int32[] = {
            0x8b, 0x0b                      // mov     ecx, DWORD PTR [rbx]
        };
    static unsigned char arg0_get_int64[] = {
            0x48, 0x8b, 0x0b                // mov     rcx, QWORD PTR [rbx]
        };
    static unsigned char arg0_get_float32[] = {
            0xf3, 0x0f, 0x10, 0x03          // movss   xmm0, DWORD PTR [rbx]
        };
    static unsigned char arg0_get_float64[] = {
            0xf2, 0x0f, 0x10, 0x03          // movsdx  xmm0, QWORD PTR [rbx]
        };

    // End ARG0 CHOICE ]]
    static unsigned char function_call[] = {
            0xff, 0xd5,                     // call    rbp
            0x49, 0x03, 0xdc                // add     rbx, r12
        };
    // Begin RESULT CHOICE [[
    static unsigned char result_set_int8[] = {
            0x88, 0x07                      // mov     BYTE PTR [rdi], al
        };
    static unsigned char result_set_int16[] = {
            0x66, 0x89, 0x07                // mov     WORD PTR [rdi], ax
        };
    static unsigned char result_set_int32[] = {
            0x89, 0x07                      // mov     DWORD PTR [rdi], eax
        };
    static unsigned char result_set_int64[] = {
            0x48, 0x89, 0x07                // mov     QWORD PTR [rdi], rax
        };
    static unsigned char result_set_float32[] = {
            0xf3, 0x0f, 0x11, 0x07          // movss   DWORD PTR [rdi], xmm0
        };
    static unsigned char result_set_float64[] = {
            0xf2, 0x0f, 0x11, 0x07          // movsdx  QWORD PTR [rdi], xmm0
        };
    // End RESULT CHOICE ]]
    static unsigned char loop_finish[] = {
            0x49, 0x03, 0xfd,               // add     rdi, r13
            0x48, 0xff, 0xce,               // dec     rsi
            0x75, 0x00                      // jne     SHORT loop_start (REQUIRES FIXUP)
        };
    // skip_loop:
    static unsigned char epilog[] = {
            0x48, 0x8b, 0x5c, 0x24, 0x40,   // mov     rbx, QWORD PTR [rsp+0x40]
            0x48, 0x8b, 0x6c, 0x24, 0x48,   // mov     rbp, QWORD PTR [rsp+0x48]
            0x48, 0x8b, 0x74, 0x24, 0x50,   // mov     rsi, QWORD PTR [rsp+0x50]
            0x48, 0x83, 0xc4, 0x20,         // add     rsp, 0x20
            0x41, 0x5d,                     // pop     r13
            0x41, 0x5c,                     // pop     r12
            0x5f,                           // pop     rdi
            0xc3                            // ret     0
        };

    // Allocate enough memory for all the variations.
    intptr_t alloc_size = sizeof(unwind_info) + sizeof(prolog) + sizeof(loop_setup) +
                        sizeof(function_call) + sizeof(loop_finish) + sizeof(epilog) + 96;
    char *code_begin, *code_current, *code_end;
    allocate_executable_memory(exec_memblock.get(), alloc_size, 16, &code_begin, &code_end);
    code_current = code_begin;

    char *loop_skip_fixup, *loop_continue_fixup;
    char *loop_start_label, *loop_skip_label;

    // The function prolog
    memcpy(code_current, prolog, sizeof(prolog));
    code_current += sizeof(prolog);

    // The loop setup
    memcpy(code_current, loop_setup, sizeof(loop_setup));
    code_current += sizeof(loop_setup);

    loop_skip_fixup = code_current - 1;
    loop_start_label = code_current;

    // Argument zero setup
    switch (arg0type.get_type_id()) {
        case bool_type_id:
        case int8_type_id:
        case uint8_type_id:
            memcpy(code_current, arg0_get_int8, sizeof(arg0_get_int8));
            code_current += sizeof(arg0_get_int8);
            break;
        case int16_type_id:
        case uint16_type_id:
            memcpy(code_current, arg0_get_int16, sizeof(arg0_get_int16));
            code_current += sizeof(arg0_get_int16);
            break;
        case int32_type_id:
        case uint32_type_id:
            memcpy(code_current, arg0_get_int32, sizeof(arg0_get_int32));
            code_current += sizeof(arg0_get_int32);
            break;
        case int64_type_id:
        case uint64_type_id:
            memcpy(code_current, arg0_get_int64, sizeof(arg0_get_int64));
            code_current += sizeof(arg0_get_int64);
            break;
        case float32_type_id:
            memcpy(code_current, arg0_get_float32, sizeof(arg0_get_float32));
            code_current += sizeof(arg0_get_float32);
            break;
        case float64_type_id:
            memcpy(code_current, arg0_get_float64, sizeof(arg0_get_float64));
            code_current += sizeof(arg0_get_float64);
            break;
        default: {
            // Get rid of what we allocated and raise an error
            resize_executable_memory(exec_memblock.get(), 0, &code_begin, &code_end);
            stringstream ss;
            ss << "The unary_kernel_adapter does not support " << arg0type << " for the argument type";
            throw runtime_error(ss.str());
        }
    }

    // The function call
    memcpy(code_current, function_call, sizeof(function_call));
    code_current += sizeof(function_call);

    // Store the return value
    switch (restype.get_type_id()) {
        case bool_type_id:
        case int8_type_id:
        case uint8_type_id:
            memcpy(code_current, result_set_int8, sizeof(result_set_int8));
            code_current += sizeof(result_set_int8);
            break;
        case int16_type_id:
        case uint16_type_id:
            memcpy(code_current, result_set_int16, sizeof(result_set_int16));
            code_current += sizeof(result_set_int16);
            break;
        case int32_type_id:
        case uint32_type_id:
            memcpy(code_current, result_set_int32, sizeof(result_set_int32));
            code_current += sizeof(result_set_int32);
            break;
        case int64_type_id:
        case uint64_type_id:
            memcpy(code_current, result_set_int64, sizeof(result_set_int64));
            code_current += sizeof(result_set_int64);
            break;
        case float32_type_id:
            memcpy(code_current, result_set_float32, sizeof(result_set_float32));
            code_current += sizeof(result_set_float32);
            break;
        case float64_type_id:
            memcpy(code_current, result_set_float64, sizeof(result_set_float64));
            code_current += sizeof(result_set_float64);
            break;
        default: {
            // Get rid of what we allocated and raise an error
            resize_executable_memory(exec_memblock.get(), 0, &code_begin, &code_end);
            stringstream ss;
            ss << "The unary_kernel_adapter does not support " << restype << " for the return type";
            throw runtime_error(ss.str());
        }
    }

    // The rest of the loop
    memcpy(code_current, loop_finish, sizeof(loop_finish));
    code_current += sizeof(loop_finish);

    loop_continue_fixup = code_current - 1;
    loop_skip_label = code_current;

    // The function epilog
    memcpy(code_current, epilog, sizeof(epilog));
    code_current += sizeof(epilog);

    char *code_function_end = code_current;

    // Apply fixups to the conditional jumps
    *loop_skip_fixup = static_cast<char>(loop_skip_label - (loop_skip_fixup + 1));
    *loop_continue_fixup = static_cast<char>(loop_start_label - (loop_continue_fixup + 1));

    // The UNWIND_INFO structure, aligned to 2 bytes
    code_current = (char *)(((uintptr_t)code_current + 0x1)&(-2));
    char *code_unwind_info = code_current;
    memcpy(code_current, unwind_info, sizeof(unwind_info));
    code_current += sizeof(unwind_info);

    // The four unary_specialization_t pointers, all point to the same function
    // because we don't specialize presently.
    code_current = (char *)(((uintptr_t)code_current + (sizeof(char *) - 1))&(-sizeof(char *)));
    char **specializations = reinterpret_cast<char **>(code_current);
    specializations[0] = code_begin;
    specializations[1] = code_begin;
    specializations[2] = code_begin;
    specializations[3] = code_begin;
    code_current += 4 * sizeof(char *);

    // Shrink the allocation to just what we needed
    resize_executable_memory(exec_memblock.get(), code_current - code_begin, &code_begin, &code_end);

    // Register the stack info so exceptions can unwind through the call
    set_executable_memory_runtime_function(exec_memblock.get(),
                    code_begin,
                    code_function_end, code_unwind_info);

    throw std::runtime_error("TODO: dynd::codegen_unary_function_adapter needs fixing for updated kernel prototype");
    //return unary_operation_pair_t(NULL, NULL);
}

#endif // defined(_WIN32) && defined(_M_X64)
