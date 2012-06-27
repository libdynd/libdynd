//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__MEMORY_BLOCK_HPP_
#define _DND__MEMORY_BLOCK_HPP_

#include <iostream>

#include <boost/detail/atomic_count.hpp>

#include <dnd/config.hpp>

#include <stdint.h>

namespace dnd {

/**
 * These are all the types of memory blocks supported by the dnd library.
 */
enum memory_block_type_t {
    /** An ndarray node is a node so that it can hold its own data memory if desired */
    ndarray_node_memory_block_type,
    /** Memory from outside the dnd library */
    external_memory_block_type,
    /** For when the data is POD and its size is fully known ahead of time */
    fixed_size_pod_memory_block_type,
    /** For when the data is POD, and the amount of memory needs to grow */
    pod_memory_block_type,
    /** For when the data is object */
    object_memory_block_type
};

/**
 * This is the data that goes at the start of every memory block, including
 * an atomic reference count and a memory_block_type_t. There is a fixed set
 * of memory block types, of which 'ndarray_node' and 'external' are the only
 * extensible ones.
 */
struct memory_block_data {
#ifdef DND_CLING
    // A hack avoiding boost atomic_count, since that creates inline assembly which LLVM JIT doesn't like!
    mutable long m_use_count;
#else
    /** Embedded reference counting using boost::intrusive_ptr */
    mutable boost::detail::atomic_count m_use_count;
#endif
    /** A memory_block_type_t enum value */
    uint32_t m_type;

    explicit memory_block_data(long use_count, memory_block_type_t type)
        : m_use_count(use_count), m_type(type)
    {
    }
};

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
    ++memblock->m_use_count;
}

/**
 * Decrements the reference count of a memory block object,
 * freeing it if the count reaches zero.
 */
inline void memory_block_decref(memory_block_data *memblock)
{
    if (--memblock->m_use_count == 0) {
        detail::memory_block_free(memblock);
    }
}

/**
 * Does a debug dump of the memory block.
 */
void memory_block_debug_dump(const memory_block_data *memblock, std::ostream& o, const std::string& indent = "");

/**
 * A smart pointer to a memory_block object. Very similar
 * to boost::intrusive_ptr<memory_block_data>.
 */
class memory_block_ref {
    memory_block_data *m_memblock;
public:
    /** Default constructor */
    memory_block_ref()
        : m_memblock(0)
    {
    }

    /** Constructor from a raw pointer */
    explicit memory_block_ref(memory_block_data *memblock, bool add_ref = true)
        : m_memblock(memblock)
    {
        if (memblock != 0 && add_ref) {
            memory_block_incref(memblock);
        }
    }

    /** Copy constructor */
    memory_block_ref(const memory_block_ref& rhs)
        : m_memblock(rhs.m_memblock)
    {
        memory_block_incref(m_memblock);
    }

#ifdef DND_RVALUE_REFS
    /** Move constructor */
    memory_block_ref(memory_block_ref&& rhs)
        : m_memblock(rhs.m_memblock)
    {
        rhs.m_memblock = 0;
    }
#endif

    /** Destructor */
    ~memory_block_ref()
    {
        if (m_memblock != 0) {
            memory_block_decref(m_memblock);
        }
    }

    /** Assignment */
    memory_block_ref& operator=(const memory_block_ref& rhs)
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
#ifdef DND_RVALUE_REFS
    memory_block_ref& operator=(memory_block_ref&& rhs)
    {
        if (m_memblock != 0) {
            memory_block_decref(m_memblock);
        }
        m_memblock = rhs.m_memblock;
        rhs.m_memblock = 0;
        return *this;
    }
#endif

    /** Gets the raw memory_block_data pointer */
    memory_block_data *get() const
    {
        return m_memblock;
    }

    /** Gives away ownership of the reference count */
    memory_block_data *release()
    {
        memory_block_data *result = m_memblock;
        m_memblock = 0;
        return result;
    }

    void swap(memory_block_ref& rhs)
    {
        memory_block_data *tmp = m_memblock;
        m_memblock = rhs.m_memblock;
        rhs.m_memblock = tmp;
    }

    bool operator==(const memory_block_ref& rhs) const
    {
        return m_memblock == rhs.m_memblock;
    }

    bool operator!=(const memory_block_ref& rhs) const
    {
        return m_memblock != rhs.m_memblock;
    }

    bool operator==(const memory_block_data *memblock) const
    {
        return m_memblock == memblock;
    }

    bool operator!=(const memory_block_data *memblock) const
    {
        return m_memblock != memblock;
    }
};

inline bool operator==(const memory_block_data *memblock, const memory_block_ref& rhs)
{
    return memblock == rhs.get();
}

inline bool operator!=(const memory_block_data *memblock, const memory_block_ref& rhs)
{
    return memblock != rhs.get();
}

typedef void (*external_memory_block_free_t)(void *);
/**
 * Creates a memory block which is a reference to an external object.
 */
memory_block_ref make_external_memory_block(void *object, external_memory_block_free_t free_fn);
/**
 * Creates a memory block of a pre-determined fixed size. A pointer to the
 * memory allocated for data is placed in the output parameter.
 */
memory_block_ref make_fixed_size_pod_memory_block(intptr_t alignment, intptr_t size, char **out_datapointer);

} // namespace dnd

#endif // _DND__MEMORY_BLOCK_HPP_
