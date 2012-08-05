
#include <dnd/platform_definitions.h>
#if defined(DND_POSIX)

// this file is only to help extract and generate the appropriate instructions
// to implement the kernels

#include <dnd/auxiliary_data.hpp>
#include <dnd/codegen/unary_kernel_adapter_codegen.hpp>


namespace dnd
{
    
typedef double TTestType;
extern "C" void disassembly_analysis (char * __restrict dst, intptr_t dst_stride,
                const char * __restrict src, intptr_t src_stride,
                intptr_t count, const AuxDataBase * __restrict auxdata)
{
    typedef TTestType (*cdecl_func_ptr_t)(TTestType);
    const unary_function_adapter_auxdata& faux = reinterpret_cast<const detail::auxiliary_data_holder<unary_function_adapter_auxdata>*>(auxdata)->get();

    cdecl_func_ptr_t kfunc = reinterpret_cast<cdecl_func_ptr_t>(faux.function_pointer);
    for (intptr_t i = 0; i < count; ++i) {
        *(TTestType *)dst = kfunc(*(const TTestType *)src);
        
        dst += dst_stride;
        src += src_stride;
    }
}


extern "C" void __attribute__((naked)) extract_opcodes()
{
    asm("movsbl     (%rbx), %edi            \n"
    "    movzbl     (%rbx), %edi            \n"
    "    movb       %al, (%rbp)             \n"
    "    movswl     (%rbx), %edi            \n"
    "    movzwl     (%rbx), %edi            \n"
    "    movw       %ax, (%rbp)             \n"
    "    movl       (%rbx), %edi            \n"
    "    movl       (%rbx), %edi            \n"
    "    movl       %eax, (%rbp)            \n"
    "    movq       (%rbx), %rdi            \n"
    "    movq       (%rbx), %rdi            \n"
    "    movq       %rax, (%rbp)            \n"
    // float
    "    movss      (%rbx), %xmm0           \n"
    "    movss      %xmm0, (%rbp)           \n"
    // double
    "    movsd      (%rbx), %xmm0           \n"
    "    movsd      %xmm0, (%rbp)           \n"
        
    );
}

extern "C" void __attribute__((naked)) test_function(char * __restrict //dst
                     , intptr_t // dst_stride
                     , const char * __restrict //src
                     , intptr_t //src_stride
                     , intptr_t //count
                     , const AuxDataBase * __restrict //auxdata
                     )
{
    asm("pushq       %rbp                    \n"
    "    pushq       %r15                    \n"
    "    pushq       %r14                    \n"
    "    pushq       %r13                    \n"
    "    pushq       %r12                    \n"
    "    pushq       %rbx                    \n"
//    "    pushq       %rax                    \n"
    
    "    movq        %r9, %r14               \n"
    "    movq        %r8, %r12               \n"
    "    movq        %rcx, %r15              \n"
    "    movq        %rdx, %rbx              \n"
    "    movq        %rsi, %r13              \n"
    "    movq        %rdi, %rbp              \n"
    
//    "    testq       %r12, %r12              \n"
//    "    jle         test_function_end_loop  \n"
    "    andq        $-2, %r14               \n"
    "    mov         32(%r14), %r14          \n"
    "    .align      4, 0x90                 \n"
    "test_function_begin_loop:               \n"
    "    movl        (%rbx), %edi            \n" //" movss (%rbx), %xmm0
    "    callq       *%r14                   \n"
    "    movl        %eax, (%rbp)            \n" //" movss (%rb
    "    addq        %r15, %rbx              \n"
    "    addq        %r13, %rbp              \n"
    "    decq        %r12                    \n"
    "    jne         test_function_begin_loop\n"
    "test_function_end_loop:                 \n"
//    "    addq        $8, %rsp                \n"
    "    popq        %rbx                    \n"
    "    popq        %r12                    \n"
    "    popq        %r13                    \n"
    "    popq        %r14                    \n"
    "    popq        %r15                    \n"
    "    popq        %rbp                    \n"
    );
}
    
} // namespace dnd
        
#endif

