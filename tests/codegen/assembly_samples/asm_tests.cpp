
#include <dnd/platform_definitions.h>
#if defined(DND_POSIX)

// this file is only to help extract and generate the appropriate instructions
// to implement the kernels

#include <dnd/auxiliary_data.hpp>
#include <dnd/codegen/unary_kernel_adapter_codegen.hpp>
#include <dnd/codegen/binary_kernel_adapter_codegen.hpp>


namespace dnd
{

template <typename TRES, typename TARG0, typename TARG1>
void disassembly_test (uint8_t* __restrict dst, ptrdiff_t dst_stride
                       , uint8_t* __restrict arg0, ptrdiff_t arg0_stride
                       , uint8_t* __restrict arg1, ptrdiff_t arg1_stride
                       , size_t count, const AuxDataBase* __restrict auxdata)
{
    typedef TRES (*cdecl_func_ptr_t)(const TARG0, const TARG1);
    const binary_function_adapter_auxdata& faux = reinterpret_cast<const detail::auxiliary_data_holder<binary_function_adapter_auxdata>*>(auxdata)->get();
    
    cdecl_func_ptr_t kfunc = reinterpret_cast<cdecl_func_ptr_t>(faux.function_pointer);
    do  {
        *(TRES *)dst = kfunc(*(const TARG0 *)arg0, *(const TARG1 *)arg1);
        
        dst += dst_stride;
        arg0 += arg0_stride;
        arg1 += arg1_stride;
    } while (--count);
}
    
template
void disassembly_test <int32_t, int32_t, int32_t>(uint8_t* __restrict dst, ptrdiff_t dst_stride
                                                  , uint8_t* __restrict arg0, ptrdiff_t arg0_stride
                                                  , uint8_t* __restrict arg1, ptrdiff_t arg1_stride
                                                  , size_t count, const AuxDataBase* __restrict auxdata);
template
void disassembly_test <uint8_t, uint8_t, uint8_t>(uint8_t* __restrict dst, ptrdiff_t dst_stride
                                                , uint8_t* __restrict arg0, ptrdiff_t arg0_stride
                                                , uint8_t* __restrict arg1, ptrdiff_t arg1_stride
                                                , size_t count, const AuxDataBase* __restrict auxdata);
    
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

    // for the binary function args are in (%r12) -arg1- or (%rbx) -arg0-
    // r12 args may go either to %edi, %rdi, %xmm0 or %xmm1 depending on the
    // types of arg0 and arg1 (in sysv the position of an argument depends on
    // the types of the arguments before it
    "    .align     4, 0x90                \n"
    "    movsbl     (%rbx), %edi            \n"
    "    .align     4, 0x90                \n"
    "    movsbl     (%r12), %edi            \n"
    "    .align     4, 0x90                \n"
    "    movsbl     (%r12), %esi            \n"

    "    .align     4, 0x90                \n"
    "    movswl     (%rbx), %edi            \n"
    "    .align     4, 0x90                \n"
    "    movswl     (%r12), %edi            \n"
    "    .align     4, 0x90                \n"
    "    movswl     (%r12), %esi            \n"

    "    .align     4, 0x90                \n"
    "    movl       (%rbx), %edi            \n"
    "    .align     4, 0x90                \n"
    "    movl       (%r12), %edi            \n"
    "    .align     4, 0x90                \n"
    "    movl       (%r12), %esi            \n"
    
    "    .align     4, 0x90                \n"
    "    movq       (%rbx), %rdi            \n"
    "    .align     4, 0x90                \n"
    "    movq       (%r12), %rdi            \n"
    "    .align     4, 0x90                \n"
    "    movq       (%r12), %rsi            \n"

    "    .align     4, 0x90                \n"
    "    movss      (%rbx), %xmm0           \n"
    "    .align     4, 0x90                \n"
    "    movss      (%r12), %xmm0           \n"
    "    .align     4, 0x90                \n"
    "    movss      (%r12), %xmm1           \n"
    
    "    .align     4, 0x90                \n"
    "    movsd      (%rbx), %xmm0           \n"
    "    .align     4, 0x90                \n"
    "    movsd      (%r12), %xmm0           \n"
    "    .align     4, 0x90                \n"
    "    movsd      (%r12), %xmm1           \n"
    "    .align     4, 0x90                \n"

    );
}
    
extern "C" void __attribute__((naked)) test_binary_function(char * __restrict //dst
                                                            , intptr_t // dst_stride
                                                            , const char * __restrict //src
                                                            , intptr_t //src_stride
                                                            , intptr_t //count
                                                            , const AuxDataBase * __restrict //auxdata
                                                            )

{
    asm("pushq       %rbp                   \n"
    "    pushq       %r15                   \n"
    "    pushq       %r14                   \n"
    "    pushq       %r13                   \n"
    "    pushq       %r12                   \n"
    "    pushq       %rbx                   \n"
    "    subq        $24, %rsp              \n" // space for 3 qwords
// pushq %rax
    
    "    movq        %r9, 16(%rsp)          \n" // arg1_stride
    "    movq        %r8, %r12              \n" // arg1
    "    movq        %rcx, 8(%rsp)          \n" // arg0_stride
    "    movq        %rdx, %rbx             \n" // arg0
    "    movq        %rsi, %r13             \n" // dst_stride
    "    movq        %rdi, %rbp             \n" // dst
    "    movq        88(%rsp), %rax         \n" // auxdata
    "    andq        $-2, %rax              \n" // mask lower bit
    "    movq        32(%rax), %r14         \n" // the function pointer
    "    movq        80(%rsp), %r15         \n" // count
    "    .align      4, 0x90                \n"
    "loop_begin:                            \n"
    "    movl        (%r12), %esi           \n" // arg1
    "    movl        (%rbx), %edi           \n" // arg0
    "    callq       *%r14                  \n" // func_call
    "    movl        %eax, (%rbp)           \n" // store result
    "    addq        16(%rsp), %r12         \n" // update arg1
    "    addq        8(%rsp), %rbx          \n" // update arg0
    "    addq        %r13, %rbp             \n" // update dst
    "    decq        %r15                   \n"
    "    jne         loop_begin             \n"
    // restore stack
    "    addq        $24, %rsp              \n"
    "    popq        %rbx                   \n"
    "    popq        %r12                   \n"
    "    popq        %r13                   \n"
    "    popq        %r14                   \n"
    "    popq        %r15                   \n"
    "    popq        %rbp                   \n"
    // naked function adds ret
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

