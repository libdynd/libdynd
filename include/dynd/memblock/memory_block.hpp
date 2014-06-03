//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__MEMORY_BLOCK_HPP_
#define _DYND__MEMORY_BLOCK_HPP_

#include <iostream>

#include <dynd/config.hpp>
#include <dynd/atomic_refcount.hpp>

namespace dynd {

/**
 * These are all the types of memory blocks supported by the dnd library.
 */
enum memory_block_type_t {
    /** A dynd array containing the arrmeta specified by the type */
    array_memory_block_type,
    /** Memory from outside the dnd library */
    external_memory_block_type,
    /** For when the data is POD and its size is fully known ahead of time */
    fixed_size_pod_memory_block_type,
    /** For when the data is POD, and the amount of memory needs to grow */
    pod_memory_block_type,
    /** Like pod_memory_block_type, but with zero-initialization */
    zeroinit_memory_block_type,
    /**
     * For when the data is object (requires destruction),
     * and the amount of memory needs to grow */
    objectarray_memory_block_type,
    /** For memory used by code generation */
    executable_memory_block_type,
    /** Wraps memory mapped files */
    memmap_memory_block_type
};

std::ostream& operator<<(std::ostream& o, memory_block_type_t mbt);

/**
 * This is the data that goes at the start of every memory block, including
 * an atomic reference count and a memory_block_type_t. There is a fixed set
 * of memory block types, of which 'external' is presently the only
 * extensible ones.
 */
struct memory_block_data {
    atomic_refcount m_use_count;
    /** A memory_block_type_t enum value */
    uint32_t m_type;

    explicit memory_block_data(long use_count, memory_block_type_t type)
        : m_use_count(use_count), m_type(type)
    {
        //std::cout << "memblock " << (void *)this << " cre: " << this->m_use_count << std::endl;
    }
};

/**
 * This is a struct of function pointers for allocating and
 * resizing POD data within a memory_block that supports it.
 */
struct memory_block_pod_allocator_api {
    /**
     * Allocates the requested amount of memory from the memory_block, returning
     * a pointer pair.
     *
     * Call this once per output variable.
     */
    void (*allocate)(memory_block_data *self, intptr_t size_bytes, intptr_t alignment, char **out_begin, char **out_end);
    /**
     * Resizes the most recently allocated memory in the memory block, updating
     * the pointer pair. This may move the memory to a new location if necessary.
     *
     * The values in inout_begin and inout_end must have been created by a
     * previous allocate() or resize() call.
     *
     * Call this to grow the memory as needed, and to trim the memory to just
     * the needed size once that is determined.
     */
    void (*resize)(memory_block_data *self, intptr_t size_bytes, char **inout_begin, char **inout_end);
    /**
     * Finalizes the memory block so it can no longer be used to allocate more
     * memory. This call may use something like realloc to try and shrink the
     * destination memory as much as possible.
     * NOTE: realloc itself may move memory for any call to it, so cannot be used
     *       (e.g. on OS X it was found that shrinking memory to 8 bytes caused it to
     *       move, likely to a special small object heap).
     */
    void (*finalize)(memory_block_data *self);
    /**
     * When a memory block is being used as a temporary buffer, resets it to
     * a state throwing away existing used memory. This allows the same memory
     * to be used for variable-sized data to be reused repeatedly in such
     * a temporary buffer.
     */
    void (*reset)(memory_block_data *self);
};

/**
 * This is a struct of function pointers for allocating
 * object data (of types with a destructor) within a
 * memory_block that supports it.
 */
struct memory_block_objectarray_allocator_api {
    /**
     * Allocates the requested amount of memory from the memory_block, returning
     * a pointer.
     *
     * Call this once per output variable.
     */
    char *(*allocate)(memory_block_data *self, size_t count);
    /**
     * Resizes the most recently allocated memory from the memory_block.
     */
    char *(*resize)(memory_block_data *self, char *previous_allocated, size_t count);
    /**
     * Finalizes the memory block so it can no longer be used to allocate more
     * memory.
     */
    void (*finalize)(memory_block_data *self);
    /**
     * When a memory block is being used as a temporary buffer, resets it to
     * a state throwing away existing used memory. This allows the same memory
     * to be used for variable-sized data to be reused repeatedly in such
     * a temporary buffer.
     */
    void (*reset)(memory_block_data *self);
};

/**
 * Returns a pointer to a static POD memory allocator API,
 * for the type of the memory block.
 */
memory_block_pod_allocator_api *get_memory_block_pod_allocator_api(memory_block_data *memblock);

/**
 * Returns a pointer to a static objectarray memory allocator API,
 * for the type of the memory block.
 */
memory_block_objectarray_allocator_api *get_memory_block_objectarray_allocator_api(memory_block_data *memblock);



namespace detail {
    /**
     * Frees the data for a memory block. Is called
     * by memory_block_decref when the reference count
     * reaches zero.
     */
    void memory_block_free(memory_block_data *memblock);
} // namespace detail

/**
 * Increments the reference count of a memory block object.
 */
inline void memory_block_incref(memory_block_data *memblock)
{
    //std::cout << "memblock " << (void *)memblock << " inc: " << memblock->m_use_count + 1 << std::endl;
    ++memblock->m_use_count;
}

/**
 * Decrements the reference count of a memory block object,
 * freeing it if the count reaches zero.
 */
inline void memory_block_decref(memory_block_data *memblock)
{
    //std::cout << "memblock " << (void *)memblock << " dec: " << memblock->m_use_count - 1 << std::endl;
    //if (memblock->m_use_count == 1) {std::cout << "memblock " << (void *)memblock << " fre: " << memblock->m_use_count - 1 << std::endl;}
    if (--memblock->m_use_count == 0) {
        detail::memory_block_free(memblock);
    }
}

/**
 * Does a debug dump of the memory block.
 */
void memory_block_debug_print(const memory_block_data *memblock, std::ostream& o, const std::string& indent = "");

/**
 * A smart pointer to a memory_block object. Very similar
 * to boost::intrusive_ptr<memory_block_data>.
 */
class memory_block_ptr {
    memory_block_data *m_memblock;
public:
    /** Default constructor */
    memory_block_ptr()
        : m_memblock(0)
    {
    }

    /** Constructor from a raw pointer */
    explicit memory_block_ptr(memory_block_data *memblock, bool add_ref = true)
        : m_memblock(memblock)
    {
        if (memblock != 0 && add_ref) {
            memory_block_incref(memblock);
        }
    }

    /** Copy constructor */
    memory_block_ptr(const memory_block_ptr& rhs)
        : m_memblock(rhs.m_memblock)
    {
        if (m_memblock != 0) {
            memory_block_incref(m_memblock);
        }
    }

#ifdef DYND_RVALUE_REFS
    /** Move constructor */
    memory_block_ptr(memory_block_ptr&& rhs)
        : m_memblock(rhs.m_memblock)
    {
        rhs.m_memblock = 0;
    }
#endif

    /** Destructor */
    ~memory_block_ptr() {
        if (m_memblock != 0) {
            memory_block_decref(m_memblock);
        }
    }

    /** Assignment */
    memory_block_ptr& operator=(const memory_block_ptr& rhs)
    {
        if (m_memblock != 0) {
            memory_block_decref(m_memblock);
        }
        if (rhs.m_memblock != 0) {
            m_memblock = rhs.m_memblock;
            memory_block_incref(m_memblock);
        } else {
            m_memblock = 0;
        }
        return *this;
    }

    /** Move assignment */
#ifdef DYND_RVALUE_REFS
    memory_block_ptr& operator=(memory_block_ptr&& rhs)
    {
        if (m_memblock != 0) {
            memory_block_decref(m_memblock);
        }
        m_memblock = rhs.m_memblock;
        rhs.m_memblock = 0;
        return *this;
    }
#endif

    /** Assignment from raw memory_block pointer */
    memory_block_ptr& operator=(memory_block_data *rhs)
    {
        if (m_memblock != 0) {
            memory_block_decref(m_memblock);
        }
        m_memblock = rhs;
        if (rhs != 0) {
            memory_block_incref(rhs);
        }
        return *this;
    }

    /** Returns true if there is only one reference to this memory block */
    bool unique() const {
        return m_memblock->m_use_count <= 1;
    }

    /** Gets the raw memory_block_data pointer */
    memory_block_data *get() const {
        return m_memblock;
    }

    /** Gives away ownership of the reference count */
    memory_block_data *release() {
        memory_block_data *result = m_memblock;
        m_memblock = 0;
        return result;
    }

    void swap(memory_block_ptr& rhs) {
        memory_block_data *tmp = m_memblock;
        m_memblock = rhs.m_memblock;
        rhs.m_memblock = tmp;
    }

    bool operator==(const memory_block_ptr& rhs) const {
        return m_memblock == rhs.m_memblock;
    }

    bool operator!=(const memory_block_ptr& rhs) const {
        return m_memblock != rhs.m_memblock;
    }

    bool operator==(const memory_block_data *memblock) const {
        return m_memblock == memblock;
    }

    bool operator!=(const memory_block_data *memblock) const {
        return m_memblock != memblock;
    }
};

inline bool operator==(const memory_block_data *memblock, const memory_block_ptr& rhs)
{
    return memblock == rhs.get();
}

inline bool operator!=(const memory_block_data *memblock, const memory_block_ptr& rhs)
{
    return memblock != rhs.get();
}

} // namespace dynd

#endif // _DYND__MEMORY_BLOCK_HPP_
