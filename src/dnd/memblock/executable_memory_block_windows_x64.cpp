//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#if defined(_WIN32) && defined(_M_X64)

#include <Windows.h>

#include <stdexcept>
#include <deque>
#include <vector>
#include <sstream>

#include <dnd/memblock/executable_memory_block.hpp>

using namespace std;
using namespace dnd;

namespace {
    struct allocated_function {
        DWORD64 m_begin, m_end;
        RUNTIME_FUNCTION m_rf;
    };

    struct virtual_alloc_chunk {
        char *m_memory_begin;
        vector<allocated_function> m_functions;
    };

    static PRUNTIME_FUNCTION get_runtime_function_callback(
                                DWORD64 ControlPc, PVOID Context)
    {
        virtual_alloc_chunk *chunk = reinterpret_cast<virtual_alloc_chunk *>(Context);
        // TODO: This isn't thread-safe, if a function is created while we're doing this, and
        //       the vector is resized, there will be problems.

        // Find the allocated function which contains the address
        for (vector<allocated_function>::iterator i = chunk->m_functions.begin(), i_end = chunk->m_functions.end();
                    i != i_end; ++i) {
            if (i->m_begin <= ControlPc && ControlPc < i->m_end) {
                return &i->m_rf;
            }
        }

        return NULL;
    }

    struct executable_memory_block {
        /** Every memory block object needs this at the front */
        memory_block_data m_mbd;
        intptr_t m_total_allocated_capacity, m_chunk_size_bytes;
        /** The VirtualAlloc'd memory, with  */
        deque<virtual_alloc_chunk> m_memory_handles;
        /** The current VirtualAlloc'd memory being doled out */
        char *m_memory_begin, *m_memory_current, *m_memory_end;

        /**
         * Allocates a new chunk of executable memory.
         */
        void append_memory()
        {
            HANDLE hProcess = GetCurrentProcess();
            m_memory_handles.push_back(virtual_alloc_chunk());
            m_memory_begin = reinterpret_cast<char *>(VirtualAllocEx(hProcess, NULL, m_chunk_size_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
            m_memory_handles.back().m_memory_begin = m_memory_begin;
            if (m_memory_begin == NULL) {
                m_memory_handles.pop_back();
                throw bad_alloc();
            }
            m_memory_current = m_memory_begin;
            m_memory_end = m_memory_current + m_chunk_size_bytes;
            m_total_allocated_capacity += m_chunk_size_bytes;

            // Install the handler for stack frame unwinding
            if (RtlInstallFunctionTableCallback((uintptr_t)m_memory_begin|0x3, (uintptr_t)m_memory_begin, m_chunk_size_bytes,
                        &get_runtime_function_callback, &m_memory_handles.back(), NULL) == 0) {
                VirtualFreeEx(hProcess, m_memory_begin, 0, MEM_RELEASE);
                m_memory_handles.pop_back();
                throw runtime_error("Error in RtlInstallFunctionTableCallback");
            }
        }

        executable_memory_block(intptr_t chunk_size_bytes)
            : m_mbd(1, executable_memory_block_type), m_total_allocated_capacity(0),
                    m_chunk_size_bytes(chunk_size_bytes), m_memory_handles()
        {
            append_memory();
        }

        ~executable_memory_block()
        {
            HANDLE hProcess = GetCurrentProcess();
            for (size_t i = 0, i_end = m_memory_handles.size(); i != i_end; ++i) {
                VirtualFreeEx(hProcess, m_memory_handles[i].m_memory_begin, 0, MEM_RELEASE);
            }
        }
    };
} // anonymous namespace

memory_block_ptr dnd::make_executable_memory_block(intptr_t chunk_size_bytes)
{
    executable_memory_block *pmb = new executable_memory_block(chunk_size_bytes);
    return memory_block_ptr(reinterpret_cast<memory_block_data *>(pmb), false);
}


namespace dnd { namespace detail {

void free_executable_memory_block(memory_block_data *memblock)
{
    executable_memory_block *emb = reinterpret_cast<executable_memory_block *>(memblock);
    delete emb;
}

void allocate_executable_memory(memory_block_data *self, intptr_t size_bytes, intptr_t alignment, char **out_begin, char **out_end)
{
//    cout << "allocating " << size_bytes << " of memory with alignment " << alignment << endl;
    // Allocate new exectuable memory of the requested size and alignment
    executable_memory_block *emb = reinterpret_cast<executable_memory_block *>(self);
    if (size_bytes > emb->m_chunk_size_bytes) {
        stringstream ss;
        ss << "Memory allocation request of " << size_bytes << " is too large for this executable_memory_block";
        ss << " with chunk size " << emb->m_chunk_size_bytes;
        throw runtime_error(ss.str());
    }
    char *begin = reinterpret_cast<char *>(
                    (reinterpret_cast<uintptr_t>(emb->m_memory_current) + alignment - 1) & ~(alignment - 1));
    char *end = begin + size_bytes;
    if (end > emb->m_memory_end) {
        // Allocate another chunk of memory
        emb->append_memory();
        begin = emb->m_memory_begin;
        end = begin + size_bytes;
    }

    // Indicate where to allocate the next memory
    emb->m_memory_current = end;

    // Return the allocated memory
    *out_begin = begin;
    *out_end = end;
//    cout << "allocated at address " << (void *)begin << endl;
}

void resize_executable_memory(memory_block_data *self, intptr_t size_bytes, char **inout_begin, char **inout_end)
{
    // Resizes previously allocated executable memory to the requested size
    executable_memory_block *emb = reinterpret_cast<executable_memory_block *>(self);
//    cout << "resizing memory " << (void *)*inout_begin << " / " << (void *)*inout_end << " from size " << (*inout_end - *inout_begin) << " to " << size_bytes << endl;
//    cout << "memory state before " << (void *)emb->m_memory_begin << " / " << (void *)emb->m_memory_current << " / " << (void *)emb->m_memory_end << endl;
    if (*inout_end != emb->m_memory_current) {
        // Simple sanity check
        throw runtime_error("executable_memory_block resize must be called only using the most recently allocated memory");
    }
    char *end = *inout_begin + size_bytes;
    if (end <= emb->m_memory_end) {
        // If it fits, just adjust the current allocation point
        emb->m_memory_current = end;
        *inout_end = end;
    } else {
        // If it doesn't fit, need to copy to a new memory chunk (note: assuming position independent code)
        char *old_begin = emb->m_memory_begin, *old_current = *inout_begin;
        // Allocate another chunk of memory
        emb->append_memory();
        memcpy(emb->m_memory_begin, *inout_begin, *inout_end - *inout_begin);
        end = emb->m_memory_begin + size_bytes;
        emb->m_memory_current = end;
        *inout_begin = emb->m_memory_begin;
        *inout_end = end;
    }
//    cout << "memory state after " << (void *)emb->m_memory_begin << " / " << (void *)emb->m_memory_current << " / " << (void *)emb->m_memory_end << endl;
}

void set_executable_memory_runtime_function(memory_block_data *self, char *begin, char *end, char *unwind_data)
{
    // Sets the runtime function info for the most recently allocated memory
    executable_memory_block *emb = reinterpret_cast<executable_memory_block *>(self);
    virtual_alloc_chunk &vac = emb->m_memory_handles.back();
    vac.m_functions.push_back(allocated_function());
    allocated_function &af = vac.m_functions.back();
    af.m_begin = reinterpret_cast<DWORD64>(begin);
    af.m_end = reinterpret_cast<DWORD64>(end);
    af.m_rf.BeginAddress = (DWORD)(begin - vac.m_memory_begin);
    af.m_rf.EndAddress = (DWORD)(end - vac.m_memory_begin);
    af.m_rf.UnwindData = (DWORD)(unwind_data - vac.m_memory_begin);
}

}} // namespace dnd::detail

void dnd::executable_memory_block_debug_dump(const memory_block_data *memblock, std::ostream& o, const std::string& indent)
{
    const executable_memory_block *emb = reinterpret_cast<const executable_memory_block *>(memblock);
    o << indent << " chunk size: " << emb->m_chunk_size_bytes << "\n";
    o << indent << " allocated: " << emb->m_total_allocated_capacity << "\n";
}


#endif // defined(_WIN32) && defined(_M_X64)