NDObject Low Level Details
==========================

This document describes low level details of DyND ndobjects, and how
they might integrate with Numba.

Memory Block
------------

#include <dynd/memblock/memory_block.hpp>

The memory block is DyND's reference counted object for storing ndobjects
and data. It always starts with two 32-bit members, an atomic use count and
a memory block type.

    struct memory_block_data {
        atomic_refcount m_use_count;
        uint32_t m_type;
    };
    
Two functions are exposed for incref/decref, which perform the atomic increment
or atomic decrement + free respectively. For integration with Numba, the increment
and decrement operations should be atomic and ideally inlined, and llvmpy has the method
Builder.atomic_add which can be used for this. When a decrement takes the use count
to zero, the function dynd::detail::memory_block_free is called to deallocate the block.
This function could be provided to Numba via a ctypes function pointer, in a 'lowlevel'
namespace of the Python exposure.

Memory Block Allocator API
--------------------------

For variable-sized blockref dtypes, a memory block which has a simple allocator
API for getting new element memory is used. This is exposed by a function
dynd::get_memory_block_pod_allocator_api which accepts a memory block pointer and
returns a pointer to a struct of three functions,

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
    };

Numba kernels which produce variable-sized strings as their output would use this API
to allocate the string output memory. A typical sequence of events for such a kernel
might be:

 * Get the memblock pointer from the string's dtype metadata.
 * Request the allocator API for that memblock pointer.
 * For each output string, first call 'allocate' to get more memory than
   needed if the amount of memory is unknown, or the exact amount if known.
   Repeatedly grow (probably double) that memory using 'resize' until the string
   is complete. Make a final call to 'resize' to shrink the allocated memory to
   just what is needed when it is complete. Set the data pointers of the output
   string to 'begin' and 'end' received from the allocator.

The same logic will apply for variable-sized arrays when they are fully implemented.

