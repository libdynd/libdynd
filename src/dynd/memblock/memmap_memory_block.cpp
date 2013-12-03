//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <limits>

#ifdef WIN32
#define NOMINMAX
#include <Windows.h>
#endif

#include <dynd/array.hpp>
#include <dynd/memblock/memmap_memory_block.hpp>

using namespace std;
using namespace dynd;

static intptr_t get_file_size(HANDLE hFile)
{
    DWORD fsLow = 0, fsHigh = 0;
    fsLow = GetFileSize(hFile, &fsHigh);
    uint64_t fs = fsLow + (static_cast<uint64_t>(fsHigh) << 32);
#ifdef _WIN64
    if (fs > (uint64_t)std::numeric_limits<intptr_t>::max()) {
        throw runtime_error("On 32-bit systems, maximum file size is 2GB");
    }
#endif
    return static_cast<intptr_t>(fs);
}

namespace {
    struct memmap_memory_block {
        /** Every memory block object needs this at the front */
        memory_block_data m_mbd;
        // Parameters used to construct the memory block
        string m_filename;
        uint32_t m_access;
        intptr_t m_begin, m_end;
        // Handle to the mapped memory
        HANDLE m_hFile, m_hMapFile;
        // Pointer to the mapped memory
        char *m_mapPointer;
        // Offset to the actual data requested (memory mapping has strict
        // alignment requirements)
        intptr_t m_mapOffset;

        memmap_memory_block(const std::string& filename,
                    uint32_t access, char **out_pointer, intptr_t *out_size,
                    intptr_t begin, intptr_t end)
            : m_mbd(1, memmap_memory_block_type), m_filename(filename),
                m_access(access), m_begin(begin), m_end(end)
        {
#ifdef WIN32
            // TODO: This function isn't quite exception-safe, use a smart pointer for the handles to fix.

            // Get the system granularity
            DWORD sysGran;
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            sysGran = sysInfo.dwAllocationGranularity;

            bool readwrite = ((access & nd::write_access_flag) == nd::write_access_flag);

            // Open the file using the windows API
            m_hFile = CreateFile(
                m_filename.c_str(),
                GENERIC_READ | (readwrite ? GENERIC_WRITE : 0),
                FILE_SHARE_READ, NULL,
                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (m_hFile == NULL) {
                stringstream ss;
                ss << "failed to open file \"" << m_filename << "\" for memory mapping";
                throw runtime_error(ss.str());
            }

            intptr_t filesize = get_file_size(m_hFile);
            // Handle the begin offset, following Python semantics of bytes[begin:end]
            if (begin < 0) {
                begin += filesize;
                if (begin < 0) {
                    begin = 0;
                }
            } else if (begin >= filesize) {
                begin = filesize;
            }
            // Handle the end offset
            if (end < 0) {
                end += filesize;
                if (end <= begin) {
                    end = begin;
                }
            } else if (end <= begin) {
                end = begin;
            } else if (end >= filesize) {
                end = filesize;
            }

            // Calculate where to to do the file mapping. It needs to be on a boundary based on the
            // system allocation granularity
            intptr_t mapbegin = (begin / sysGran) * sysGran;
            m_mapOffset = begin - mapbegin;
            intptr_t mapsize = end - mapbegin;

            m_hMapFile = CreateFileMapping(m_hFile, NULL,
                readwrite ? PAGE_READWRITE : PAGE_READONLY,
#ifdef _WIN64
                (uint32_t)(((uint64_t)end) >> 32),
#else
                0,
#endif
                (uint32_t)end,
                NULL);
            if (m_hMapFile == NULL) {
                CloseHandle(m_hFile);
                stringstream ss;
                ss << "failure mapping file \"" << m_filename << "\" for memory mapping";
                throw runtime_error(ss.str());
            }

            // Create the mapped memory
            m_mapPointer = (char *)MapViewOfFile(
                m_hMapFile,
                FILE_MAP_READ | (readwrite ? FILE_MAP_WRITE : 0),
#ifdef _WIN64
                (uint32_t)(((uint64_t)mapbegin) >> 32),
#else
                0,
#endif
                (uint32_t)mapbegin,
                mapsize);
            if (m_mapPointer == NULL) {
                CloseHandle(m_hMapFile);
                CloseHandle(m_hFile);
                stringstream ss;
                ss << "failure mapping view of file \"" << m_filename << "\" for memory mapping";
                throw runtime_error(ss.str());
            }
            *out_pointer = m_mapPointer + m_mapOffset;
            *out_size = end - begin;
#else // Finished win32 implementation, now posix
            throw runtime_error("TODO: implement posix memmap");
#endif
        }

        ~memmap_memory_block()
        {
#ifdef WIN32
            UnmapViewOfFile(m_mapPointer);
            CloseHandle(m_hMapFile);
            CloseHandle(m_hFile);
#else
            throw runtime_error("TODO: implement posix memmap");
#endif
        }
    };
} // anonymous namespace

memory_block_ptr dynd::make_memmap_memory_block(const std::string& filename,
    uint32_t access, char **out_pointer, intptr_t *out_size,
    intptr_t begin, intptr_t end)
{
    memmap_memory_block *pmb = new memmap_memory_block(
        filename, access, out_pointer, out_size, begin, end);
    return memory_block_ptr(reinterpret_cast<memory_block_data *>(pmb), false);
}

namespace dynd { namespace detail {

void free_memmap_memory_block(memory_block_data *memblock)
{
    memmap_memory_block *emb = reinterpret_cast<memmap_memory_block *>(memblock);
    delete emb;
}

}} // namespace dynd::detail

void dynd::memmap_memory_block_debug_print(const memory_block_data *memblock, std::ostream& o, const std::string& indent)
{
    const memmap_memory_block *emb = reinterpret_cast<const memmap_memory_block *>(memblock);
    o << indent << " filename: " << emb->m_filename << "\n";
    o << indent << " begin: " << emb->m_begin << "\n";
    o << indent << " end: " << emb->m_end << "\n";
}
