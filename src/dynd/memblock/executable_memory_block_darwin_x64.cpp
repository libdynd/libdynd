//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//


// To enable logging just uncomment this:
//#define ENABLE_LOGGING


#include <dynd/platform_definitions.hpp>
#if defined(DYND_OS_DARWIN)

#include <dynd/memblock/executable_memory_block.hpp>

// system includes
#include <sys/mman.h>

// standard includes
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <assert.h>
#include <errno.h>

/*
static inline bool ptr_in_range(void* ptr, void* lower, void* upper)
{
    return (ptr >= lower) && (ptr < upper);    
}
*/

static inline void* ptr_offset(void* base, ptrdiff_t offset_in_bytes)
{
    return static_cast<void*>(static_cast<int8_t*>(base) + offset_in_bytes);
}

static inline size_t align_up(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}


////////////////////////////////////////////////////////////////////////////////
namespace {

struct executable_memory_block : public dynd::memory_block_data 
{
    executable_memory_block(size_t chunk_size_in_bytes);
    ~executable_memory_block();
    
    void add_chunk();
    
    size_t              m_chunk_size;       // maximum chunk size
    void*               m_pivot;
    std::vector<void*>  m_allocated_chunks;    
};

executable_memory_block::executable_memory_block(size_t chunk_size_in_bytes)
    : dynd::memory_block_data(1, dynd::executable_memory_block_type)
    , m_chunk_size(align_up(chunk_size_in_bytes, getpagesize()))
{
}

executable_memory_block::~executable_memory_block()
{
    std::vector<void*>::const_iterator curr = m_allocated_chunks.begin();
    std::vector<void*>::const_iterator end = m_allocated_chunks.end();
    while (curr < end)
    {
        munmap(*curr, m_chunk_size);
        ++curr;
    }
    
    // not needed, just to leave no traces behind :)
    m_allocated_chunks.clear();
    m_chunk_size = 0;
    m_pivot = 0;
}
    
void executable_memory_block::add_chunk()
{
    void* result = mmap(0  // no address hint
                        , m_chunk_size // the size
                        , PROT_READ | PROT_WRITE | PROT_EXEC // rwx
                        , MAP_PRIVATE | MAP_ANON
                        , 0, 0);
    
    if (result != MAP_FAILED)
    {
        m_allocated_chunks.push_back(result);
        m_pivot = result;
    }
    else 
    {
        std::stringstream ss;
        ss << "mmap failed with errno = " << errno << ": "<< strerror(errno);
        throw std::runtime_error(ss.str());
    }
}

} // nameless namespace

////////////////////////////////////////////////////////////////////////////////

namespace dynd { 
    
namespace detail {
void free_executable_memory_block(memory_block_data *memblock)
{
    executable_memory_block *emb = static_cast<executable_memory_block *>(memblock);
    delete emb;
}
} // namespace detail


memory_block_ptr make_executable_memory_block(intptr_t chunk_size_bytes)
{
    executable_memory_block *pmb = new executable_memory_block(chunk_size_bytes);
    return memory_block_ptr(pmb, false);
}
    
void allocate_executable_memory(memory_block_data * self       //in
                                , intptr_t          size_bytes //in
                                , intptr_t          alignment  //in
                                , char **           out_begin  //out
                                , char **           out_end    //out
                                )
{
    executable_memory_block* emb = static_cast<executable_memory_block*>(self);
    // some preconditions
    assert( executable_memory_block_type == executable_memory_block_type);
    assert( (size_t)size_bytes <= emb->m_chunk_size );
#ifdef ENABLE_LOGGING
    std::cout << "allocating " << size_bytes 
              << " of executable memory with alignment " << alignment 
              << std::endl;
#endif //ENABLE LOGGING
    
    if ((size_t)size_bytes > emb->m_chunk_size)
    {
        std::stringstream ss;
        ss << "Memory allocation request of " << size_bytes 
           << " is too large for this executable_memory_block"
              " with chunk size" << emb->m_chunk_size;
        throw std::runtime_error(ss.str());
    }
    
    if (emb->m_allocated_chunks.empty())
        emb->add_chunk();
    
    void* current_chunk = emb->m_allocated_chunks.back();
    void* begin = reinterpret_cast<void*>(align_up(reinterpret_cast<size_t>(emb->m_pivot), alignment));
    void* end   = ptr_offset(begin, size_bytes);
    if (ptr_offset(current_chunk, emb->m_chunk_size) < ptr_offset(emb->m_pivot, size_bytes))
    {
        emb->add_chunk();
        begin = emb->m_allocated_chunks.back();
        end   = ptr_offset(begin, size_bytes);
    }

    emb->m_pivot = end;
    /*
    assert(ptr_in_range(begin
                        , emb->m_allocated_chunks.back()
                        , ptr_offset(emb->m_allocated_chunks.back()
                                     , emb->m_chunk_size)));
    assert(ptr_in_range(end
                        , emb->m_allocated_chunks.back()
                        , ptr_offset(emb->m_allocated_chunks.back()
                        , emb->m_chunk_size)));
    */
    assert(((int8_t*)end - (int8_t*)begin) == size_bytes);
    assert(emb->m_pivot == end);
    *out_begin = static_cast<char*>(begin);
    *out_end = static_cast<char*>(end);
    
}

void resize_executable_memory(memory_block_data * self
                            , intptr_t            new_size
                            , char **             inout_begin
                            , char **             inout_end
                            )
{
    executable_memory_block* emb = static_cast<executable_memory_block*>(self);
    void* current_chunk = emb->m_allocated_chunks.back();
    
    void* old_begin = static_cast<void*>(*inout_begin);
    void* old_end   = static_cast<void*>(*inout_end);
    
    assert(old_end == emb->m_pivot);

    void* new_begin = old_begin;
    void* new_end   = ptr_offset(old_begin, new_size);

    if (new_end >= ptr_offset(current_chunk, emb->m_chunk_size))
    {
        emb->add_chunk();
        new_begin = emb->m_allocated_chunks.back();
        new_end   = ptr_offset(new_begin, new_size);
        size_t old_size = static_cast<uint8_t*>(old_end) - static_cast<uint8_t*>(old_begin);
        memcpy(new_begin, old_begin, old_size);
        *inout_begin = static_cast<char*>(new_begin);
    }
 
    emb->m_pivot = new_end;
    *inout_end   = static_cast<char*>(new_end);

}
    
void executable_memory_block_debug_print(const memory_block_data *memblock
                                        , std::ostream& os
                                        , const std::string& indent
                                        )
{
    const executable_memory_block *emb = static_cast<const executable_memory_block *>(memblock);
    size_t chunk_size = emb->m_chunk_size;
    void*  current_chunk = emb->m_allocated_chunks.back();
    ptrdiff_t current_chunk_used_bytes = static_cast<uint8_t*>(emb->m_pivot) - static_cast<uint8_t*>(current_chunk);
    size_t allocated  = emb->m_allocated_chunks.size() * (chunk_size - 1)
                        + current_chunk_used_bytes;
    os << indent << " chunk size: " << chunk_size << std::endl;
    os << indent << " allocated: "  << allocated << std::endl;
    os << indent << " system page size: " << getpagesize() << std::endl;

    /*
    for (size_t i = 0, i_end = emb->m_memory_handles.size(); i != i_end; ++i) {
        const virtual_alloc_chunk& vac = emb->m_memory_handles[i];
        os << indent << " allocated chunk at address " << (void *)vac.m_memory_begin << ":\n";
        for (size_t j = 0, j_end = vac.m_functions.size(); j != j_end; ++j) {
            const RUNTIME_FUNCTION &rf = vac.m_functions[j];
            os << indent << "  RUNTIME_FUNCTION{" << (void *)rf.BeginAddress;
            os << ", " << (void *)rf.EndAddress << ", " << (void *)rf.UnwindData << "}\n";
        }
    }
    */
}
} // namespace dynd

#endif // defined(_WIN32) && defined(_M_X64)
