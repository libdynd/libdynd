//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

#ifdef WIN32
inline intptr_t get_file_size(HANDLE hFile) {
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
#endif

inline void clip_begin_end(intptr_t size, intptr_t &begin, intptr_t &end) {
  if (begin < 0) {
    begin += size;
    if (begin < 0) {
      begin = 0;
    }
  } else if (begin >= size) {
    begin = size;
  }
  // Handle the end offset
  if (end < 0) {
    end += size;
    if (end <= begin) {
      end = begin;
    }
  } else if (end <= begin) {
    end = begin;
  } else if (end >= size) {
    end = size;
  }
}

struct memmap_memory_block : public memory_block_data {
  // Parameters used to construct the memory block
  std::string m_filename;
  uint32_t m_access;
  intptr_t m_begin, m_end;
// Handle to the mapped memory
#ifdef WIN32
  HANDLE m_hFile, m_hMapFile;
#else
  int m_fd;
#endif
  // Pointer to the mapped memory
  char *m_mapPointer;
  // Offset to the actual data requested (memory mapping has strict
  // alignment requirements)
  intptr_t m_mapOffset;

  memmap_memory_block(const std::string &filename, uint32_t access, char **out_pointer, intptr_t *out_size,
                      intptr_t begin, intptr_t end)
      : memory_block_data(1, memmap_memory_block_type), m_filename(filename), m_access(access), m_begin(begin),
        m_end(end) {
    bool readwrite = false; // ((access & nd::write_access_flag) == nd::write_access_flag);
#ifdef WIN32
    // TODO: This function isn't quite exception-safe, use a smart pointer for the handles to fix.

    // Get the system granularity
    DWORD sysGran;
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    sysGran = sysInfo.dwAllocationGranularity;

    // Open the file using the windows API
    m_hFile = CreateFile(m_filename.c_str(), GENERIC_READ | (readwrite ? GENERIC_WRITE : 0), FILE_SHARE_READ, NULL,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m_hFile == NULL) {
      std::stringstream ss;
      ss << "failed to open file \"" << m_filename << "\" for memory mapping";
      throw std::runtime_error(ss.str());
    }

    intptr_t filesize = get_file_size(m_hFile);
    // Handle the begin offset, following Python
    // semantics of bytes[begin:end]
    clip_begin_end(filesize, begin, end);
    m_begin = begin;
    m_end = end;

    // Calculate where to to do the file mapping. It needs to be
    // on a boundary based on the system allocation granularity
    intptr_t mapbegin = (begin / sysGran) * sysGran;
    m_mapOffset = begin - mapbegin;
    intptr_t mapsize = end - mapbegin;

    m_hMapFile = CreateFileMapping(m_hFile, NULL, readwrite ? PAGE_READWRITE : PAGE_READONLY,
#ifdef _WIN64
                                   (uint32_t)(((uint64_t)end) >> 32),
#else
                                   0,
#endif
                                   (uint32_t)end, NULL);
    if (m_hMapFile == NULL) {
      CloseHandle(m_hFile);
      std::stringstream ss;
      ss << "failure mapping file \"" << m_filename << "\" for memory mapping";
      throw std::runtime_error(ss.str());
    }

    // Create the mapped memory
    m_mapPointer = (char *)MapViewOfFile(m_hMapFile, FILE_MAP_READ | (readwrite ? FILE_MAP_WRITE : 0),
#ifdef _WIN64
                                         (uint32_t)(((uint64_t)mapbegin) >> 32),
#else
                                         0,
#endif
                                         (uint32_t)mapbegin, mapsize);
    if (m_mapPointer == NULL) {
      CloseHandle(m_hMapFile);
      CloseHandle(m_hFile);
      std::stringstream ss;
      ss << "failure mapping view of file \"" << m_filename << "\" for memory mapping";
      throw std::runtime_error(ss.str());
    }
    *out_pointer = m_mapPointer + m_mapOffset;
    *out_size = end - begin;
#else // Finished win32 implementation, now posix
    m_fd = open(m_filename.c_str(), readwrite ? O_RDWR : O_RDONLY);
    if (m_fd == -1) {
      std::stringstream ss;
      ss << "failed to open file \"" << m_filename << "\" for memory mapping";
      throw std::runtime_error(ss.str());
    }
#ifdef __APPLE__
    // [From the Python mmap code] Issue #11277:
    // fsync(2) is not enough on OS X - a special, OS X specific
    // fcntl(2) is necessary to force DISKSYNC and get around
    // mmap(2) bug
    (void)fcntl(m_fd, F_FULLFSYNC);
#endif
    struct stat st;
    if (fstat(m_fd, &st) == -1) {
      std::stringstream ss;
      ss << "failed to stat file \"" << m_filename << "\" for memory mapping";
      throw std::runtime_error(ss.str());
    }
    intptr_t filesize = st.st_size;

    // Handle the begin offset, following Python
    // semantics of bytes[begin:end]
    clip_begin_end(filesize, begin, end);
    m_begin = begin;
    m_end = end;

    intptr_t pageSize = sysconf(_SC_PAGE_SIZE);
    intptr_t mapbegin = (begin / pageSize) * pageSize;
    m_mapOffset = begin - mapbegin;
    intptr_t mapsize = end - mapbegin;

    m_mapPointer = (char *)mmap(NULL, mapsize, PROT_READ | (readwrite ? PROT_WRITE : 0), MAP_SHARED, m_fd, mapbegin);
    if (m_mapPointer == (char *)MAP_FAILED) {
      close(m_fd);
      std::stringstream ss;
      ss << "failed to mmap file \"" << m_filename << "\" for memory mapping";
      throw std::runtime_error(ss.str());
    }

    *out_pointer = m_mapPointer + m_mapOffset;
    *out_size = end - begin;
#endif
  }

  ~memmap_memory_block() {
#ifdef WIN32
    UnmapViewOfFile(m_mapPointer);
    CloseHandle(m_hMapFile);
    CloseHandle(m_hFile);
#else
    intptr_t mapsize = m_end - m_begin + m_mapOffset;
    munmap((void *)m_mapPointer, mapsize);
    close(m_fd);
#endif
  }

  void debug_print(std::ostream &o, const std::string &indent);
};

/**
 * Creates a memory block of a memory-mapped file.
 *
 * \param filename  The filename of the file to memory map.
 * \param access  A combination of write_access_flag, read_access_flag, immutable_access_flag.
 * \param out_pointer  This is the pointer to the mapped memory.
 * \param out_size  This is the size of the mapped memory. Note that the size may be different
 *                  than requested by begin/end, because this function uses Python semantics to
 *                  clip out of bounds boundaries to the data.
 * \param begin  The position within the file to start the memory map
 *               (default beginning of the file). This value may be
 *               negative, in which case it is interpreted as an offset from the
 *               end of the file.
 * \param end  The position within the file to end the memory map
 *             (default end of the file). This value may be
 *             negative, in which case it is interpreted as an offset from the
 *             end of the file.
 */
DYNDT_API intrusive_ptr<memory_block_data>
make_memmap_memory_block(const std::string &filename, uint32_t access, char **out_pointer, intptr_t *out_size,
                         intptr_t begin = 0, intptr_t end = std::numeric_limits<intptr_t>::max());
} // namespace dynd
