//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//


#include <dnd/platform_definitions.h>

#if defined(DND_CALL_SYSV_X64)

#include <dnd/codegen/unary_kernel_adapter_codegen.hpp>
#include <dnd/memblock/executable_memory_block.hpp>
#include <dnd/memblock/memory_block.hpp>
#include <dnd/dtype.hpp>
#include <sstream>
#include <stdexcept>


namespace
{
    void* ptr_offset(void* ptr, ptrdiff_t offset)
    {
        return static_cast<void*>(static_cast<uint8_t*>(ptr) + offset);
    }
    
    const char* type_to_str(unsigned restype)
    {
        switch (restype)
        {
            case 0: return "int8";
            case 1: return "int16";
            case 2: return "int32";
            case 3: return "int64";
            case 4: return "float32";
            case 5: return "float64";
            default: return "unknown type";
        }
    }
    
    
    size_t idx_for_type_id(dnd::type_id_t type_id)
    {
        // this is related to the array of code snippets below... handle with
        // care
        using namespace dnd;
        switch (type_id)
        {
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
        default:
            {
                std::stringstream ss;
                ss << "unary kernel adapter does not support " << dtype(type_id) << " as return type";
                throw ( ss.str() );
            }
        }
    }
}

namespace dnd
{


uint64_t get_unary_function_adapter_unique_id(const dtype& restype
                                              , const dtype& arg0type
                                              , calling_convention_t DND_UNUSED(callconv)
                                              )
{
    // Bits 0..2 for the result type
    uint64_t result = idx_for_type_id(restype.type_id());
    
    // Bits 3..5 for the arg0 type
    result += idx_for_type_id(arg0type.type_id()) << 3;
    
    // There is only one calling convention on Windows x64, so it doesn't
    // need to get encoded in the unique id.
    
    return result;    
}
    
    
std::string get_unary_function_adapter_unique_id_string(uint64_t unique_id)
{
    std::stringstream ss;

    const char* str_ret = type_to_str(unique_id & 0xf);
    const char* str_arg = type_to_str((unique_id>>4) &0xf);
    ss << str_ret << " (" << str_arg << ")";
    return ss.str();
}

namespace // nameless
{
// these are portions of machine code used to compose the unary function adapter    
    static unsigned char unary_adapter_prolog[] = {
        // save callee saved registers... we use them all ;)
        0x55,                           // pushq %rbp
        0x41, 0x57,                     // pushq %r15
        0x41, 0x56,                     // pushq %r14
        0x41, 0x55,                     // pushq %r13
        0x41, 0x54,                     // pushq %r12
        0x54,                           // pushq %rbx
    };
    
    static unsigned char unary_adapter_loop_setup[] = {
        // move all the registers to callee saved ones... we are reusing them
        // in the loop
        0x4d, 0x89, 0xce,               // movq %r9, %r14
        0x4d, 0x89, 0xc4,               // movq %r8, %r12
        0x49, 0x89, 0xcf,               // movq %rcx, %r15
        0x48, 0x89, 0xd3,               // movq %rdx, %rbx
        0x49, 0x89, 0xf5,               // movq %rsi, %r13
        0x48, 0x89, 0xfd,               // movq %rdi, %rbp
        // and mask the lower bit on what is the function kernel (0xfe == -2)
        0x49, 0x83, 0xe6, 0xfe,         // andq $0xfe, %r14
        // then fetch the actual function pointer..
        0x4d, 0x8b, 0x76, 0x20,         // movq 0x20(%r14), %r14
    };

    
    static unsigned char unary_adapter_arg0_get_int8[] = {
        0x0f, 0xbe, 0x3b,               // movsbl   (%rbx), %edi
    };
    static unsigned char unary_adapter_arg0_get_int16[] = {
        0x0f, 0xbf, 0x3b,               // movswl   (%rbx), %edi
    };
    static unsigned char unary_adapter_arg0_get_int32[] = {
        0x8b, 0x3b                      // movl     (%rbx), %edi
    };
    static unsigned char unary_adapter_arg0_get_int64[] = {
        0x48, 0x8b, 0x3b                // movq     (%rbx), %rdi
    };
    static unsigned char unary_adapter_arg0_get_float32[] = {
        0xf3, 0x0f, 0x10, 0x03          // movss    (%rbx), %xmm0
    };
    static unsigned char unary_adapter_arg0_get_float64[] = {
        0xf2, 0x0f, 0x10, 0x03          // movsd    (%rbx), %xmm0
    };

    // End ARG0 CHOICE ]]
    static unsigned char unary_adapter_function_call[] = {
        // our function pointer is in %r14
        0x41, 0xff, 0xd6,               // call     *%r14
    };
    // Begin RESULT CHOICE [[
    static unsigned char unary_adapter_result_set_int8[] = {
        0x88, 0x45, 0x00                // movb     %al, 0x00(%rbp)
    };
    static unsigned char unary_adapter_result_set_int16[] = {
        0x66, 0x89, 0x45, 0x00          // movw     %ax, 0x00(%rbp)
    };
    static unsigned char unary_adapter_result_set_int32[] = {
        0x89, 0x45, 0x00                // movl     %eax, (%rbp)
    };
    static unsigned char unary_adapter_result_set_int64[] = {
        0x48, 0x89, 0x45, 0x00          // moq      %rax, (%rbp)
    };
    static unsigned char unary_adapter_result_set_float32[] = {
        0xf3, 0x0f, 0x11, 0x45, 0x00    // movss    %xmm0, (%rbp)
    };
    static unsigned char unary_adapter_result_set_float64[] = {
        0xf2, 0x0f, 0x11, 0x45, 0x00    // movsd    %xmm0, (%rbp)
    };
    // End RESULT CHOICE ]]
    static unsigned char unary_adapter_update_streams[] = {
        0x4c, 0x01, 0xfb,               // add  %r15, %rbx  # update src
        0x4c, 0x01, 0xed,               // addq %r13, %rbp  # update dst
        0x49, 0xff, 0xcc,               // decq %r12        # dec count
        0x75, 0x00,                     // jne loop (patch last byte)
    };
    // skip_loop:
    static unsigned char unary_adapter_epilog[] = {
        // restore callee saved registers and return...
        0x5b,                           // popq %rbx
        0x41, 0x5c,                     // popq %r12
        0x41, 0x5d,                     // popq %r13
        0x41, 0x5e,                     // popq %r14
        0x41, 0x5f,                     // popq %r15
        0x5d,                           // popq %rbp
        0xc3,                           // ret
    };
    
    typedef struct { void* ptr; size_t size; } _code_snippet;
    static _code_snippet arg0_snippets[] =
    {
        { unary_adapter_arg0_get_int8,            sizeof(unary_adapter_arg0_get_int8)            },
        { unary_adapter_arg0_get_int16,           sizeof(unary_adapter_arg0_get_int16)           },
        { unary_adapter_arg0_get_int32,           sizeof(unary_adapter_arg0_get_int32)           },
        { unary_adapter_arg0_get_int64,           sizeof(unary_adapter_arg0_get_int64)           },
        { unary_adapter_arg0_get_float32,         sizeof(unary_adapter_arg0_get_float32)         },
        { unary_adapter_arg0_get_float64,         sizeof(unary_adapter_arg0_get_float64)         },
    };
    
    static _code_snippet ret_snippets[] =
    {
        { unary_adapter_result_set_int8,            sizeof(unary_adapter_result_set_int8)        },
        { unary_adapter_result_set_int16,           sizeof(unary_adapter_result_set_int16)       },
        { unary_adapter_result_set_int32,           sizeof(unary_adapter_result_set_int32)       },
        { unary_adapter_result_set_int64,           sizeof(unary_adapter_result_set_int64)       },
        { unary_adapter_result_set_float32,         sizeof(unary_adapter_result_set_float32)     },
        { unary_adapter_result_set_float64,         sizeof(unary_adapter_result_set_float64)     },
    };
    
    
    // function_builder is a helper to generate machine code for our adapters.
    // It copies snippets of code and allows appending empty space (filled with
    // nops) as well as having some "label" and alignment support. The label
    // support is based on offsets so that we may support relocating the code
    // (as long as the generated code is PIC or the required fixups are
    // implemented).
    class function_builder
    {
    public:
        function_builder(memory_block_data* memblock, size_t estimated_size);
        ~function_builder();
        
        function_builder& label(size_t& where);

        function_builder& append(const void* code, size_t code_size);
        function_builder& append(size_t code_size);
        function_builder& align (size_t alignment);
        
        void*             base() const;
        bool              is_ok() const;
        
        void              finish();
        void              discard();
        
    private:
        memory_block_data*  memblock_;
        int8_t*             current_;
        int8_t*             begin_;
        int8_t*             end_;
        bool                ok_;
    };

    function_builder::function_builder(memory_block_data* memblock, size_t estimated_size)
        : memblock_(memblock)
        , current_(0)
        , begin_(0)
        , end_(0)
        , ok_(true)
    {
        dnd::allocate_executable_memory(memblock, estimated_size, 16, (char**)&begin_, (char**)&end_);
        current_ = begin_;
    }
    
    function_builder::~function_builder()
    {
        discard();
    }
    
    function_builder& function_builder::label(size_t& offset)
    {
        offset = current_ - begin_;
        return *this;
    }
    
    function_builder& function_builder::append(const void* code, size_t code_size)
    {
        if (!ok_)
            return *this;
        
        assert(end_ >= current_);
        if (static_cast<size_t>(end_ - current_) >= code_size)
        {
            memcpy(current_, code, code_size);
            current_ += code_size;
        }
        else
            ok_ = false;
        
        return *this;
    }
    
    function_builder& function_builder::append(size_t code_size)
    {
        if (!ok_)
            return *this;
        
        assert(end_ >= current_);
        if (static_cast<size_t>(end_ - current_) >= code_size)
        {
            // fill with nop (0x90)
            memset(current_, 0x90, code_size);
            current_ += code_size;
        }
        else
            ok_ = false;
        
        return *this;
    }
    
    function_builder& function_builder::align(size_t code_size)
    {
        // align to a given size.
        intptr_t modulo = reinterpret_cast<uintptr_t>(current_) % code_size;
        if (0 != modulo)
            append(code_size - modulo);
        
        assert(0 == reinterpret_cast<uintptr_t>(current_) % code_size);
        return *this;
    }
    
    void* function_builder::base() const
    {
        return static_cast<void*>(begin_);
    }
    
    bool function_builder::is_ok() const
    {
        return ok_;
    }
    
    void function_builder::finish()
    {
        assert(is_ok());
        if (is_ok())
        {
            // this will shrink... resize_executable_memory has realloc semantics
            // so on shrink it will never move;
#ifndef _NDEBUG
            int8_t* old_begin = begin_;
#endif
            dnd::resize_executable_memory(memblock_, current_ - begin_, (char**)&begin_, (char**)&end_);
            assert(old_begin = begin_);

            // TODO: flush instruction cache for the generated code. Not needed
            //       on intel architectures, but a function placeholder if we
            //       factor this code out could be worth if we end supporting
            //       other platforms (both ARM and PPC require explicit i-cache
            //       flushing.

            
            // now mark the object as released...
            memblock_= 0;
            begin_   = 0;
            end_     = 0;
            current_ = 0;
            ok_      = false;
        }
    }
    
    void function_builder::discard()
    {
        if (begin_)
        {
            // if we didn't finish... rollback the executable memory
            dnd::resize_executable_memory(memblock_, 0, (char**)&begin_, (char**)&end_);
        } 
        memblock_ = 0;
        current_  = 0;
        begin_    = 0;
        end_      = 0;
        ok_       = false;
    }
} // nameless namespace
    

unary_operation_t* codegen_unary_function_adapter(const memory_block_ptr& exec_mem_block
                                                 , const dtype& restype
                                                 , const dtype& arg0type
                                                 , calling_convention_t DND_UNUSED(callconv))    
{
    size_t arg0_idx = idx_for_type_id(arg0type.type_id());
    size_t ret_idx  = idx_for_type_id(restype.type_id());
    
    if (arg0_idx >= sizeof(arg0_snippets)/sizeof(arg0_snippets[0]) 
        || ret_idx >= sizeof(ret_snippets)/sizeof(ret_snippets[0]))
    {
        return 0;
    }
    
    // an (over)estimation of the size of the generated function. 64 is an 
    // overestimation of the code that gets chosen based on args and ret value.
    // that is 
    size_t estimated_size = sizeof(unary_adapter_prolog)
                          + sizeof(unary_adapter_loop_setup) 
                          + sizeof(unary_adapter_function_call)
                          + sizeof(unary_adapter_update_streams)
                          + sizeof(unary_adapter_epilog) 
                          + 64; 

    size_t entry_point= 0;
    size_t loop_start = 0;
    size_t loop_end   = 0;
    size_t table      = 0;
    function_builder fbuilder(exec_mem_block.get(), estimated_size);
    fbuilder.label(entry_point)
            .append(unary_adapter_prolog, sizeof(unary_adapter_prolog))
            .append(unary_adapter_loop_setup, sizeof(unary_adapter_loop_setup))
            .align(4)
            .label(loop_start)
            .append(arg0_snippets[arg0_idx].ptr, arg0_snippets[arg0_idx].size)
            .append(unary_adapter_function_call, sizeof(unary_adapter_function_call))
            .append(ret_snippets[ret_idx].ptr, ret_snippets[ret_idx].size)
            .append(unary_adapter_update_streams, sizeof(unary_adapter_update_streams))
            .label(loop_end)
            .append(unary_adapter_epilog, sizeof(unary_adapter_epilog))
            .align(4)
            .label(table)
            .append(sizeof(unary_operation_t)*4);
                 
    if (fbuilder.is_ok())
    {
        // apply fix-ups. The fix-ups needed are for offsets in the loop skip
        // and the loop close locations
        int loop_size = loop_end - loop_start;
        void* base    = fbuilder.base();

        assert(loop_size > 0 && loop_size < 128);
        int8_t* loop_continue_offset = static_cast<int8_t*>(ptr_offset(base, loop_end)) - 1;
        *loop_continue_offset = - loop_size;
        
        unary_operation_t* specializations = static_cast<unary_operation_t*>(ptr_offset(base, table));
        unary_operation_t func_ptr = reinterpret_cast<unary_operation_t>(ptr_offset(base, entry_point));
        
        for (int i = 0; i < 4; ++i)
            specializations[i] = func_ptr;
        fbuilder.finish();
        
        return specializations;
    }
    
    // function construction failed.. fbuilder destructor will take care of
    // releasing memory (it acts as RAII, kind of -- exception safe as well)
    return 0;
}
} // namespace dnd

#endif // defined(DND_CALL_SYSV_X64)