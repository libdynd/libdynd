//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//


#include <dynd/platform_definitions.hpp>

#if 0 // (temporarily disabled) defined(DYND_CALL_SYSV_X64)

#include <sstream>
#include <stdexcept>
#include <cassert>

#include <dynd/codegen/unary_kernel_adapter_codegen.hpp>
#include <dynd/memblock/executable_memory_block.hpp>
#include <dynd/memblock/memory_block.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

namespace
{
    void* ptr_offset(void* ptr, ptrdiff_t offset)
    {
        return static_cast<void*>(static_cast<char*>(ptr) + offset);
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
    
    
    size_t idx_for_type_id(dynd::type_id_t type_id)
    {
        // this is related to the array of code snippets below... handle with
        // care
        using namespace dynd;
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
                throw std::runtime_error( ss.str() );
            }
        }
    }
    
    // function_builder is a helper to generate machine code for our adapters.
    // It copies snippets of code and allows appending empty space (filled with
    // nops) as well as having some "label" and alignment support. The label
    // support is based on offsets so that we may support relocating the code
    // (as long as the generated code is PIC or the required fixups are
    // implemented).
    class function_builder
    {
        // Make it non-copyable
        function_builder(const function_builder&);
        function_builder& operator=(const function_builder&);
    public:
        function_builder(dynd::memory_block_data* memblock, size_t estimated_size);
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
        dynd::memory_block_data*  memblock_;
        char*             current_;
        char*             begin_;
        char*             end_;
        bool                ok_;
    };
    
    function_builder::function_builder(dynd::memory_block_data* memblock,
                                       size_t estimated_size)
    : memblock_(memblock)
    , current_(0)
    , begin_(0)
    , end_(0)
    , ok_(true)
    {
        dynd::allocate_executable_memory(memblock,
                                         estimated_size,
                                         16,
                                         &begin_,
                                         &end_
                                        );
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
    
    function_builder& function_builder::append(const void* code,
                                                size_t code_size
                                               )
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
#if !defined(NDEBUG)
            char* old_begin = begin_;
#endif
            dynd::resize_executable_memory(memblock_, current_ - begin_, &begin_, &end_);
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
            dynd::resize_executable_memory(memblock_, 0, &begin_, &end_);
        }
        memblock_ = 0;
        current_  = 0;
        begin_    = 0;
        end_      = 0;
        ok_       = false;
    }
    
}

uint64_t dynd::get_unary_function_adapter_unique_id(const ndt::type& restype,
                                               const ndt::type& arg0type,
                                               calling_convention_t DYND_UNUSED(callconv)
                                              )
{
    // Bits 0..2 for the result type
    uint64_t result = idx_for_type_id(restype.get_type_id());
    
    // Bits 3..5 for the arg0 type
    result += idx_for_type_id(arg0type.get_type_id()) << 3;
    
    // There is only one calling convention on Windows x64, so it doesn't
    // need to get encoded in the unique id.
    
    return result;    
}
    
    
std::string dynd::get_unary_function_adapter_unique_id_string(uint64_t unique_id)
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
    uint8_t unary_adapter_prolog[] = {
        // save callee saved registers... we use them all ;)
        0x55,                           // pushq %rbp
        0x41, 0x57,                     // pushq %r15
        0x41, 0x56,                     // pushq %r14
        0x41, 0x55,                     // pushq %r13
        0x41, 0x54,                     // pushq %r12
        0x53,                           // pushq %rbx
    };
    
    uint8_t unary_adapter_loop_setup[] = {
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

    
    uint8_t unary_adapter_arg0_get_int8[] = {
        0x0f, 0xbe, 0x3b,               // movsbl   (%rbx), %edi
    };
    uint8_t unary_adapter_arg0_get_int16[] = {
        0x0f, 0xbf, 0x3b,               // movswl   (%rbx), %edi
    };
    uint8_t unary_adapter_arg0_get_int32[] = {
        0x8b, 0x3b                      // movl     (%rbx), %edi
    };
    uint8_t unary_adapter_arg0_get_int64[] = {
        0x48, 0x8b, 0x3b                // movq     (%rbx), %rdi
    };
    uint8_t unary_adapter_arg0_get_float32[] = {
        0xf3, 0x0f, 0x10, 0x03          // movss    (%rbx), %xmm0
    };
    uint8_t unary_adapter_arg0_get_float64[] = {
        0xf2, 0x0f, 0x10, 0x03          // movsd    (%rbx), %xmm0
    };

    // End ARG0 CHOICE ]]
    uint8_t unary_adapter_function_call[] = {
        // our function pointer is in %r14
        0x41, 0xff, 0xd6,               // call     *%r14
    };
    // Begin RESULT CHOICE [[
    uint8_t unary_adapter_result_set_int8[] = {
        0x88, 0x45, 0x00                // movb     %al, 0x00(%rbp)
    };
    uint8_t unary_adapter_result_set_int16[] = {
        0x66, 0x89, 0x45, 0x00          // movw     %ax, 0x00(%rbp)
    };
    uint8_t unary_adapter_result_set_int32[] = {
        0x89, 0x45, 0x00                // movl     %eax, (%rbp)
    };
    uint8_t unary_adapter_result_set_int64[] = {
        0x48, 0x89, 0x45, 0x00          // moq      %rax, (%rbp)
    };
    uint8_t unary_adapter_result_set_float32[] = {
        0xf3, 0x0f, 0x11, 0x45, 0x00    // movss    %xmm0, (%rbp)
    };
    uint8_t unary_adapter_result_set_float64[] = {
        0xf2, 0x0f, 0x11, 0x45, 0x00    // movsd    %xmm0, (%rbp)
    };
    // End RESULT CHOICE ]]
    uint8_t unary_adapter_update_streams[] = {
        0x4c, 0x01, 0xfb,               // addq %r15, %rbx  # update src
        0x4c, 0x01, 0xed,               // addq %r13, %rbp  # update dst
    };
    
    uint8_t unary_adapter_loop_finish[] = {
        0x49, 0xff, 0xcc,               // decq %r12        # dec count
        0x75, 0x00,                     // jne loop (patch last byte)
    };
 
    // skip_loop:
    uint8_t unary_adapter_epilog[] = {
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
    _code_snippet arg0_snippets[] =
    {
        { unary_adapter_arg0_get_int8,            sizeof(unary_adapter_arg0_get_int8)            },
        { unary_adapter_arg0_get_int16,           sizeof(unary_adapter_arg0_get_int16)           },
        { unary_adapter_arg0_get_int32,           sizeof(unary_adapter_arg0_get_int32)           },
        { unary_adapter_arg0_get_int64,           sizeof(unary_adapter_arg0_get_int64)           },
        { unary_adapter_arg0_get_float32,         sizeof(unary_adapter_arg0_get_float32)         },
        { unary_adapter_arg0_get_float64,         sizeof(unary_adapter_arg0_get_float64)         },
    };
    
    _code_snippet ret_snippets[] =
    {
        { unary_adapter_result_set_int8,            sizeof(unary_adapter_result_set_int8)        },
        { unary_adapter_result_set_int16,           sizeof(unary_adapter_result_set_int16)       },
        { unary_adapter_result_set_int32,           sizeof(unary_adapter_result_set_int32)       },
        { unary_adapter_result_set_int64,           sizeof(unary_adapter_result_set_int64)       },
        { unary_adapter_result_set_float32,         sizeof(unary_adapter_result_set_float32)     },
        { unary_adapter_result_set_float64,         sizeof(unary_adapter_result_set_float64)     },
    };
} // nameless namespace
    

unary_operation_pair_t dynd::codegen_unary_function_adapter(const memory_block_ptr& exec_mem_block,
                                                  const ndt::type& restype,
                                                  const ndt::type& arg0type,
                                                  calling_convention_t DYND_UNUSED(callconv)
                                                 )
{
    size_t arg0_idx = idx_for_type_id(arg0type.get_type_id());
    size_t ret_idx  = idx_for_type_id(restype.get_type_id());

    if (arg0_idx >= sizeof(arg0_snippets)/sizeof(arg0_snippets[0])
        || ret_idx >= sizeof(ret_snippets)/sizeof(ret_snippets[0]))
    {
        return unary_operation_pair_t();
    }

    // an (over)estimation of the size of the generated function. 64 is an
    // overestimation of the code that gets chosen based on args and ret value.
    // that is
    size_t estimated_size = sizeof(unary_adapter_prolog)
                          + sizeof(unary_adapter_loop_setup)
                          + sizeof(unary_adapter_function_call)
                          + sizeof(unary_adapter_loop_finish)
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
            .label(loop_start)
            .append(arg0_snippets[arg0_idx].ptr, arg0_snippets[arg0_idx].size)
            .append(unary_adapter_function_call, sizeof(unary_adapter_function_call))
            .append(ret_snippets[ret_idx].ptr, ret_snippets[ret_idx].size)
            .append(unary_adapter_update_streams, sizeof(unary_adapter_update_streams))
            .append(unary_adapter_loop_finish, sizeof(unary_adapter_loop_finish))
            .label(loop_end)
            .append(unary_adapter_epilog, sizeof(unary_adapter_epilog))
            .align(4)
            .label(table);

    if (fbuilder.is_ok())
    {
        // fix-up the offset of the jump closing the loop
        int loop_size = loop_end - loop_start;
        void* base    = fbuilder.base();

        assert(loop_size > 0 && loop_size < 128);
        char* loop_continue_offset = static_cast<char*>(ptr_offset(base, loop_end)) - 1;
        *loop_continue_offset = - loop_size;
        
        //unary_single_operation_deprecated_t func_ptr = reinterpret_cast<unary_single_operation_deprecated_t>(ptr_offset(base, entry_point));

        fbuilder.finish();

        throw std::runtime_error("TODO: dynd::codegen_unary_function_adapter needs fixing for updated kernel prototype");
        //return specializations;
    }

    // function construction failed.. fbuilder destructor will take care of
    // releasing memory (it acts as RAII, kind of -- exception safe as well)
    return unary_operation_pair_t();
}

#endif // defined(DYND_CALL_SYSV_X64)
