//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cassert>

#include <dynd/platform_definitions.hpp>

#if defined(DYND_CALL_SYSV_X64)

#include <dynd/codegen/binary_kernel_adapter_codegen.hpp>
#include <dynd/memblock/executable_memory_block.hpp>

#include <assert.h>

using namespace std;

#if 0
namespace // nameless
{
    void* ptr_offset(void* ptr, std::ptrdiff_t offset)
    {
        return static_cast<void*>(static_cast<char*>(ptr) + offset);
    }
    
    // ov: todo: stuff in this nameless namespace is replicated in unary_kernel_adapter...
    // explicitly assigned values. They are used as indices.
    enum cc_register_class {
        ccrc_integer_8bit  = 0,
        ccrc_integer_16bit = 1,
        ccrc_integer_32bit = 2,
        ccrc_integer_64bit = 3,
        ccrc_float_32bit   = 4,
        ccrc_float_64bit   = 5,
        ccrc_unknown       = 6,
        ccrc_count         = ccrc_unknown
    };

    const char* type_to_str(cc_register_class rc)
    {
        switch (rc)
        {
            case ccrc_integer_8bit: return "int8";
            case ccrc_integer_16bit: return "int16";
            case ccrc_integer_32bit: return "int32";
            case ccrc_integer_64bit: return "int64";
            case ccrc_float_32bit: return "float32";
            case ccrc_float_64bit: return "float64";
            default: return "unknown type";
        }
    }
    
    bool is_float_register_class(cc_register_class rc)
    {
        return ((ccrc_float_32bit == rc) || (ccrc_float_64bit == rc));
    }
    cc_register_class idx_for_type_id(dynd::type_id_t type_id)
    {
        // related with the array of code snippets... handle with care
        using namespace dynd;
        switch (type_id) {
            case bool_type_id:
            case int8_type_id:
            case uint8_type_id:
                return ccrc_integer_8bit;
            case int16_type_id:
            case uint16_type_id:
                return ccrc_integer_16bit;
            case int32_type_id:
            case uint32_type_id:
                return ccrc_integer_32bit;
            case int64_type_id:
            case uint64_type_id:
                return ccrc_integer_64bit;
            case float32_type_id:
                return ccrc_float_32bit;
            case float64_type_id:
                return ccrc_float_64bit;
            default: {
                std::stringstream ss;
                ss << "The binary_kernel_adapter does not support " << ndt::type(type_id) << " for the return type";
                throw std::runtime_error(ss.str());
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
    public:
        function_builder(dynd::memory_block_data* memblock, size_t estimated_size);
        ~function_builder();
        
        function_builder& label(size_t& where);
        
        function_builder& append(const void* code, size_t code_size);
        function_builder& append(size_t code_size);
        function_builder& align (size_t alignment);
        
        function_builder& add_argument(cc_register_class type_id_idx);
        function_builder& add_result(cc_register_class type_id_idx);
        
        void*             base() const;
        bool              is_ok() const;
        
        void              finish();
        void              discard();
        
    private:
        void              emit (char byte);
        
        // non copyable 
        function_builder(function_builder&);
        function_builder& operator= (const function_builder&);


        dynd::memory_block_data*  memblock_;
        char*             current_;
        char*             begin_;
        char*             end_;
        int8_t              current_arg_;
        bool                used_float_arg_;
        bool                ok_;
    };
    
    
    function_builder::function_builder(dynd::memory_block_data* memblock
                                       , size_t estimated_size)
    : memblock_(memblock)
    , current_(0)
    , begin_(0)
    , end_(0)
    , current_arg_(0)
    , used_float_arg_(false)
    , ok_(true)
    {
        dynd::allocate_executable_memory(memblock
                                        , estimated_size
                                        , 16
                                        , &begin_
                                        , &end_
                                        );
        current_ = begin_;
    }
    
    function_builder::~function_builder()
    {
        discard();
    }
    
    void function_builder::emit(char byte)
    {
        ok_ = ok_ && (current_ < end_);
        if (!ok_)
            return; // no space
      
        *current_++ = byte;
    }
    
    function_builder& function_builder::label(size_t& offset)
    {
        offset = current_ - begin_;
        return *this;
    }
    
    function_builder& function_builder::append(const void* code
                                               , size_t code_size
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
    
    function_builder& function_builder::add_argument(cc_register_class rc)
    {
        // this one is complex... depending on the arg number we will use either
        // %rbx (arg0) or %r12 (arg1) as stream pointer.
        // the target register depends on the type register class and, in the
        // case of arg1, in the register class for arg0...
        if (current_arg_ > 1)
        {
            ok_ = false;
        }
        
        if (ok_)
        {
            if (current_arg_ == 0)
            {
                switch (rc)
                {
                case ccrc_integer_8bit:
                    emit(0x0f); emit(0xbe); emit(0x3b);
                    break;
                case ccrc_integer_16bit:
                    emit(0x0f); emit(0xbf); emit(0x3b);
                    break;
                case ccrc_integer_32bit:
                    emit(0x8b); emit(0x3b);
                    break;
                case ccrc_integer_64bit:
                    emit(0x48); emit(0x8b); emit(0x3b);
                    break;
                case ccrc_float_32bit:
                    emit(0xf3); emit(0x0f); emit(0x10); emit(0x03);
                    break;
                case ccrc_float_64bit:
                    emit(0xf2); emit(0x0f); emit(0x10); emit(0x03);
                    break;
                default:
                    throw std::runtime_error("internal error");
                }
                
                used_float_arg_ = is_float_register_class(rc);
            }
            else
            {
                // some magick...
                char magick_mask = (used_float_arg_ == is_float_register_class(rc)) ? 0x08 : 0x00;
                switch (rc)
                {
                case ccrc_integer_8bit:
                    emit(0x41); emit(0x0f); emit(0xbe); emit(0x3c ^ magick_mask); emit(0x24);
                    break;
                case ccrc_integer_16bit:
                    emit(0x41); emit(0x0f); emit(0xbf); emit(0x3c ^ magick_mask); emit(0x24);
                    break;
                case ccrc_integer_32bit:
                    emit(0x41); emit(0x8b); emit(0x3c ^ magick_mask); emit(0x24);
                    break;
                case ccrc_integer_64bit:
                    emit(0x49); emit(0x8b); emit(0x3c ^ magick_mask); emit(0x24);
                    break;
                case ccrc_float_32bit:
                    emit(0xf3); emit(0x41); emit(0x0f); emit(0x10); emit(0x04 ^ magick_mask); emit(0x24);
                    break;
                case ccrc_float_64bit:
                    emit(0xf2); emit(0x41); emit(0x0f); emit(0x10); emit(0x04 ^ magick_mask); emit(0x24);
                    break;
                default:
                    throw std::runtime_error("internal error");
                }
                
            }
            current_arg_++;
        }
        
        return *this;
    }
    
    function_builder& function_builder::add_result(cc_register_class rc)
    {
        switch(rc)
        {
        case ccrc_integer_8bit:
            emit(0x88);
            break;
        case ccrc_integer_16bit:
            emit(0x66); emit(0x89);
            break;
        case ccrc_integer_32bit:
            emit(0x89);
            break;
        case ccrc_integer_64bit:
            emit(0x48); emit(0x89);
            break;
        case ccrc_float_32bit:
            emit(0xf3); emit(0x0f); emit(0x11);
            break;
        case ccrc_float_64bit:
            emit(0xf2); emit(0x0f); emit(0x11);
            break;
        default:
                // should never happen
                throw std::runtime_error("internal error");
        }
        // common tail meaning "register 0 of a kin - rax/xmm0"
        emit(0x45);
        emit(0x00);
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
#ifndef NDEBUG
            char* old_begin = begin_; // only used in asserts
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
    
} // anonymous namespace


uint64_t dynd::get_binary_function_adapter_unique_id(const ndt::type& restype
                                               , const ndt::type& arg0type
                                               , const ndt::type& arg1type
                                               , calling_convention_t DYND_UNUSED(callconv)
                                               )
{
    // Bits 0..2 for the result type
    uint64_t result = idx_for_type_id(restype.get_type_id());
    
    // Bits 3..5 for the arg0 type
    result += idx_for_type_id(arg0type.get_type_id()) << 3;
    
    // Bits 6..8 for the arg0 type
    result += idx_for_type_id(arg1type.get_type_id()) << 6;
    
    // There is only one calling convention on Windows x64, so it doesn't
    // need to get encoded in the unique id.
    
    return result;
}

std::string dynd::get_binary_function_adapter_unique_id_string(uint64_t unique_id)
{
    std::stringstream ss;
    ss << type_to_str(cc_register_class(unique_id & 0x07)) << " ("
    << type_to_str(cc_register_class((unique_id >> 3) & 0x07)) << ", "
    << type_to_str(cc_register_class((unique_id >> 6) & 0x07)) << ")";
    return ss.str();
}
    
    
namespace // nameless
{
// snippets of code to generate the machine code for the adapter
    uint8_t binary_adapter_prolog[] =
    {
        // save callee saved registers... we use them all ;)
        0x55,                           // pushq %rbp
        0x41, 0x57,                     // pushq %r15
        0x41, 0x56,                     // pushq %r14
        0x41, 0x55,                     // pushq %r13
        0x41, 0x54,                     // pushq %r12
        0x53,                           // pushq %rbx
        0x48, 0x83, 0xec, 0x18,         // subq  $24, %rsp
    };
    
    uint8_t binary_adapter_loop_setup[] =
    {
        0x4c, 0x89, 0x4c, 0x24, 0x10,   // movq %r9, 16(%rsp)
        0x4d, 0x89, 0xc4,               // movq %r8, %r12
        0x48, 0x89, 0x4c, 0x24, 0x08,   // movq %rcx, 8(%rsp)
        0x48, 0x89, 0xd3,               // movq %rdx, %rbx
        0x49, 0x89, 0xf5,               // movq %rsi, %r13
        0x48, 0x89, 0xfd,               // movq %rdi, %rbp
        0x48, 0x8b, 0x44, 0x24, 0x58,   // movq 88(%rsp), %rax
        0x48, 0x83, 0xe0, 0xfe,         // andq $-2, %rax
        0x4c, 0x8b, 0x70, 0x20,         // movq 32(%rax), %r14
        0x4c, 0x8b, 0x7c, 0x24, 0x50,   // movq 80(%rsp), %r15
    };
    
    uint8_t binary_adapter_function_call[] =
    {
        // function pointer in %r14
        0x41, 0xff, 0xd6,               // call *%r14
    };
    
    uint8_t binary_adapter_update_streams[] =
    {
        // strides are in %r13, 8(%rsp), 16(%rsp)
        0x4c, 0x03, 0x64, 0x24, 0x10,   // addq 16(%rsp),%r12
        0x48, 0x03, 0x5c, 0x24, 0x08,   // addq 8(%rsp),%rbx
        0x4c, 0x01, 0xed                // addq    %r13,%rbp

    };
    uint8_t binary_adapter_close_loop[] =
    {
        0x49, 0xff, 0xcf,               // decq %r15
        0x75, 0x00,                     // jne loop, needs fix-up
    };
    
    uint8_t binary_adapter_epilog[] =
    {
        // restore callee saved registers and return...
        0x48, 0x83, 0xc4, 0x18,         // addq $24, %rsp
        0x5b,                           // popq %rbx
        0x41, 0x5c,                     // popq %r12
        0x41, 0x5d,                     // popq %r13
        0x41, 0x5e,                     // popq %r14
        0x41, 0x5f,                     // popq %r15
        0x5d,                           // popq %rbp
        0xc3,                           // ret
    };
} // anonymous namespace

dynd::binary_operation_pair_t dynd::codegen_binary_function_adapter(const memory_block_ptr& exec_memblock
                                                        , const ndt::type& restype
                                                        , const ndt::type& arg0type
                                                        , const ndt::type& arg1type
                                                        , calling_convention_t DYND_UNUSED(callconv)
                                                        )
{
    cc_register_class ret_idx  = idx_for_type_id(restype.get_type_id());
    cc_register_class arg0_idx = idx_for_type_id(arg0type.get_type_id());
    cc_register_class arg1_idx = idx_for_type_id(arg1type.get_type_id());
    
    if ( (arg0_idx >= ccrc_count)
        || (arg1_idx >= ccrc_count)
        || (ret_idx >= ccrc_count)
        )
    {
        return binary_operation_pair_t();
    }
        
    size_t estimated_size = sizeof(binary_adapter_prolog)
                            + sizeof(binary_adapter_loop_setup)
                            + sizeof(binary_adapter_function_call)
                            + sizeof(binary_adapter_update_streams)
                            + sizeof(binary_adapter_epilog)
                            + 64;
 
    size_t entry_point= 0;
    size_t loop_start = 0;
    size_t loop_end   = 0;

    // ov: no specialization table in binary_adaptor
    function_builder fbuilder(exec_memblock.get(), estimated_size);
    fbuilder.label(entry_point)
            .append(binary_adapter_prolog, sizeof(binary_adapter_prolog))
            .append(binary_adapter_loop_setup, sizeof(binary_adapter_loop_setup))
            .label(loop_start)
            .add_argument(arg0_idx)
            .add_argument(arg1_idx)
            .append(binary_adapter_function_call, sizeof(binary_adapter_function_call))
            .add_result(ret_idx)
            .append(binary_adapter_update_streams, sizeof(binary_adapter_update_streams))
            .append(binary_adapter_close_loop, sizeof(binary_adapter_close_loop))
            .label(loop_end)
            .append(binary_adapter_epilog, sizeof(binary_adapter_epilog));
    
    if (fbuilder.is_ok())
    {
        // fix-up the offset of the jump closing the loop
        int loop_size = loop_end - loop_start;
        void* base = fbuilder.base();
        
        assert(loop_size > 0 && loop_size < 128);
        int8_t* loop_close_offset = static_cast<int8_t*>(ptr_offset(base, loop_end)) - 1;
        *loop_close_offset = - loop_size;
                
        fbuilder.finish();
        
        throw runtime_error("FIXME: codegen function return");
        //return reinterpret_cast<binary_operation_t>(ptr_offset(base, entry_point));
    }
    
    // function construction failed... fbuilder destructor will take care of
    // releasing memory (it acts as RAII, kind of -- exception safe as well)
    return binary_operation_pair_t();
}
#endif // if 0

#endif // DYND_CALL_SYSV_X64
