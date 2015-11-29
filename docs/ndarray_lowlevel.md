ND::Array Low Level Details
===========================

This document describes low level details of DyND nd::arrays, and how
they might integrate with Numba.

Memory Block
------------

    #include <dynd/memblock/memory_block.hpp>

The memory block is DyND's reference counted object for storing nd::arrays
and data. It always starts with two 32-bit members, an atomic use count and
a memory block type.

    struct memory_block_data {
        atomic_refcount m_use_count;
        uint32_t m_type;
    };

Two functions are exposed for incref/decref, which perform the atomic
increment or atomic decrement + free respectively. For integration
with Numba, the increment and decrement operations should be atomic
and ideally inlined, and llvmpy has the method Builder.atomic_add
which can be used for this. When a decrement takes the use count
to zero, the function dynd::detail::memory_block_free is called to
deallocate the block. This function could be provided to Numba via a
ctypes function pointer, in a 'lowlevel' namespace of the Python exposure.

Memory Block Allocator API
--------------------------

For variable-sized blockref dtypes, a memory block which has a simple
allocator API for getting new element memory is used. This is exposed
by a function dynd::get_memory_block_pod_allocator_api which accepts
a memory block pointer and returns a pointer to a struct of
three functions,

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

Numba kernels which produce variable-sized strings as their output
would use this API to allocate the string output memory.
A typical sequence of events for such a kernel might be:

 * Get the memblock pointer from the string's dtype arrmeta.
 * Request the allocator API for that memblock pointer.
 * For each output string, first call 'allocate' to get more memory than
   needed if the amount of memory is unknown, or the exact amount if known.
   Repeatedly grow (probably double) that memory using 'resize' until the string
   is complete. Make a final call to 'resize' to shrink the allocated memory to
   just what is needed when it is complete. Set the data pointers of the output
   string to 'begin' and 'end' received from the allocator.

The same logic will apply for variable-sized arrays when they are fully implemented.

ND::Array
---------

    #include <dynd/array.hpp>

The nd::array is a specific type of memory block, with its 'type' set to
the value array_memory_block_type (from dynd/memblock/memory_block.hpp).
Its primary structure is defined (in dynd/memblock/array_memory_block.hpp)
by the struct dynd::array_preamble. This struct looks like

    struct array_preamble {
        memory_block_data m_memblockdata;
        const base_type *m_type;
        char *m_data_pointer;
        uint64_t m_flags;
        memory_block_data *m_data_reference;
    };

This struct begins with the standard memory block data (use count
and type), then has a reference to a dtype, a pointer to the data,
and a small bit of metadata that exists in all nd::arrays,
including some flags and a memblock reference for the data.
Presently the flags are only used for access control (read,
write, and immutable).

Many dtypes require additional metadata, called arrmeta, and this is
stored in memory immediately after the dynd::array_preamble. The dtype has
a method get_arrmeta_size() which returns how much memory is needed here.

NDT::Type
---------

    #include <dynd/type.hpp>

DTypes are represented in one of two ways. The most basic types,
like the ones built into C/C++, are classified as "builtin dtypes",
and represented simply by an enumeration value. This means there are
no atomic reference increments/decrements when these dtypes are
passed around, and no memory is ever allocated or freed for them.
More complex types are subclasses of the virtual base class
base_dtype, which defines the interface for dtypes.

To distinguish between these two cases, there is a constant
`builtin_type_id_mask`, which indicates the bits used for the
builtin type id. This is used by the dtype.is_builtin()
method to test whether only the lowest bits are set.

The interface to a dtype is defined by virtual functions in the base
class extended_dtype. These functions can be broadly grouped into
three different kinds, functions which operate on just dtypes,
functions which use the dtype and arrmeta together, and functions
which use dtype, arrmeta, and data together.

This API is presently designed for use from C++, and with some
stabilization of the fundamental dtypes for arrays and various
primitives, an important development will be to create a C API
that wraps the C++ implementation. Whether and how to create a mechanism
to create new dtypes in other languages like C or Python is
something for later.

One likely development path is to restructure the extended_dtype type
hierarchy so that is has a predictable memory layout irrespective
of the C++ compiler it is created with. The tricky part of this is
to maintain the convenience, performance, and brevity of C++ virtual
functions in creating such a scheme.


### Example: Accessing a One-dimensional Strided Array

The structure of a simple 1D strided array is as follows, as seen from Python:

    >>> a = nd.array([1, 2, 3, 4])
    >>> nd.debug_repr(a)
    ------ array
     address: 00000000003FAE30
     refcount: 1
     type:
      pointer: 000007FEE194E8F0
      type: strided * int32
     arrmeta:
      flags: 5 (read_access immutable )
      type-specific arrmeta:
       strided_dim arrmeta
        stride: 4
        size: 4
     data:
       pointer: 00000000003FAE70
       reference: 0000000000000000 (embedded in array memory)
    ------

    >>> nd.type_of(a)
    ndt.type('strided * int32')

    >>> nd.dtype_of(a)
    ndt.int32

If you have a raw pointer to this object, you can first check its dtype
by inspecting it as a pointer to `dynd::array_preamble`, whose structure
is given above. After checking that the type_id of the type is
`strided_dim_type_id`, the type can be cast to a strided_dim_type pointer.

