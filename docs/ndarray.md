The DyND ND::Array
==================

The DyND nd::array is a multidimensional data storage container, inspired
by the NumPy ndarray and based on the Blaze datashape system. Like NumPy,
it supports strided multidimensional arrays of data with a uniform
data type, but has the ability to store ragged arrays and data types
with variable-sized data.

ND::Array Structure
-------------------

The inspiration for the data structure is to break apart the NumPy
ndarray into three components, a data type, some metadata like strides
and shape, and the data. Here's how this looks in NumPy:

    >>> a = np.array([[1,2,3],[4,5,6]])
    # data type
    >>> a.dtype
    dtype('int32')
    # metadata
    >>> a.shape
    (2L, 3L)
    >>> a.strides
    (12L, 4L)
    >>> a.flags
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      UPDATEIFCOPY : False
    # data
    >>> a.data
    <read-write buffer for 0x00000000028131B0, size 24, offset 0 at 0x00000000031F0FB8>

In the DyND nd::array's Python exposure, the same data is:

    >>> a = nd.array([[1,2,3],[4,5,6]])
    >>> nd.debug_repr(a)
    ------ array
     address: 00000000003F6B50
     refcount: 1
     type:
      pointer: 000000000041F8B0
      type: strided * strided * int32
     arrmeta:
      flags: 5 (read_access immutable )
      type-specific arrmeta:
       strided_dim arrmeta
        stride: 12
        size: 2
        strided_dim arrmeta
         stride: 4
         size: 3
     data:
       pointer: 00000000003F6BA0
       reference: 0000000000000000 (embedded in array memory)
    ------


Some things that were metadata in NumPy ndarrays have become part of
the dtype in DyND nd::arrays. The fact that this is a two-dimensional
strided array is encoded in the DyND dtype. One way to think of this
dtype is that it is a strided array of strided int32 arrays.

In the debug printout of the nd::array, there is first some metadata
about the nd::array (called arrmeta), some access permission flags.
The following dtype-specific arrmeta is determined by the structure of
the dtype. In this case, each strided_array part of the dtype owns
some arrmeta memory which contains its stride and dimension size.
The int32 doesn't require any additional arrmeta.

The data consists of a pointer to the memory containing the array elements,
and a reference to a memory block which holds the data
for the nd::array. In this case, the data has been embedded within
the same memory allocation as the arrmeta.

Memory Blocks
-------------

Data for nd::arrays is stored in memory blocks, which are low level
reference counted objects containing allocated memory or a
handle/reference from some other system. For example, when a NumPy ndarray
is converted into an nd::array, a PyObject pointer and a version of
Py_DECREF which grabs the GIL is held by the memory block.

The memory block itself doesn't know where within it the data an
nd::array needs is, so whereever nd::arrays need memory block references,
they also need a raw data pointer.

### Indexing Example

Here's a small example showing the result of a simple
indexing operation.

    >>> a = nd.array([1,2,3])
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
        size: 3
     data:
       pointer: 00000000003FAE70
       reference: 0000000000000000 (embedded in array memory)
    ------


    >>> nd.debug_repr(a[1])
    ------ array
     address: 0000000000415480
     refcount: 1
     type:
      pointer: 0000000000000004
      type: int32
     arrmeta:
      flags: 5 (read_access immutable )
     data:
       pointer: 00000000003FAE74
       reference: 00000000003FAE30
        ------ memory_block at 00000000003FAE30
         reference count: 2
         type: array
         type: strided * int32
        ------
    ------


In the printout of a[1], the first thing to note is the
dtype pointer, it's just the value 4. This is because
for a small number of builtin dtypes, their dtype representation
in the nd::array is just a type id.

Compare the pointer and reference of a[1] with that of a.
The pointer is 4 greater, as expected for indexing element 1 with
a stride of 4. The reference is the same as the nd::array a's
address, which you can see at the top of the printout. That's because
the array data was embedded in the nd::array's memory, so a reference
to that nd::array gets substituted for NULL while indexing.

### NumPy Example

Here's an example of an array sourced from NumPy. To make it
more interesting, we cause the memory of the array to be unaligned.

    >>> mem = np.zeros(9, dtype='i1')
    >>> a = mem[1:].view(dtype='i4')
    >>> b = nd.view(a)
    >>> nd.debug_repr(b)
    ------ array
     address: 000000000041F9A0
     refcount: 1
     type:
      pointer: 000000000041F950
      type: strided * unaligned[int32]
     arrmeta:
      flags: 3 (read_access write_access )
      type-specific arrmeta:
       strided_dim arrmeta
        stride: 4
        size: 2
     data:
       pointer: 0000000003BC4A71
       reference: 00000000004078B0
        ------ memory_block at 00000000004078B0
         reference count: 1
         type: external
         object void pointer: 00000000043DD670
         free function: 000007FEE15E2743
        ------
    ------

Because the data isn't aligned, the nd::array3 can't have a straight
int32 dtype. The solution is to make an unaligned[int32]
dtype, which has alignment 1 instead of alignment 4 like int32.

The data reference is an external memory block here. The "object void pointer"
is a pointer to the PyObject*, and the "free function" is a function which wraps
Py_DECREF in some code to ensure the GIL is being held. Unfortunately, this means
that freeing this object will be more expensive than normal, but there isn't really
another option that permits nd::arrays to be used safely across multiple threads.
For memory blocks themselves, an atomic increment/decrement is used to provide this
thread safety.

### Variable-Length String Example

The default string dtype for dynd is parameterized by its encoding and is
variable-length.  This is handled by having a memory block reference in the
string's arrmeta, and the primary string data itself being a pair of pointers
to the beginning and one past the end of the string. For an example, here's
how a single Python string converts to an nd::array:

    >>> a = nd.array("This is a string")
    >>> nd.debug_repr(a)
    ------ array
     address: 00000000003FAE30
     refcount: 1
     type:
      pointer: 00000000004154C0
      type: string
     arrmeta:
      flags: 5 (read_access immutable )
      type-specific arrmeta:
       string arrmeta
        ------ NULL memory block
     data:
       pointer: 00000000003FAE68
       reference: 0000000000000000 (embedded in array memory)
    ------


What you will notice here is that the memory block inside of the
string arrmeta is NULL, as is the data reference at the nd::array
level. This is because a larger amount of memory was allocated for
the nd::array, and the space at the end was used for the string,
to minimize the number of memory allocations. Both of the NULL
memory block references indicate this.

Let's do this also for an array of strings::

    >>> a = nd.array([u"This", u"is", u"unicode."])
    >>> nd.debug_repr(a)
    ------ array
     address: 000000000041FEC0
     refcount: 1
     type:
      pointer: 000000000041F9F0
      type: strided * string
     arrmeta:
      flags: 5 (read_access immutable )
      type-specific arrmeta:
       strided_dim arrmeta
        stride: 16
        size: 3
        string arrmeta
         ------ memory_block at 000000000041FA40
          reference count: 1
          type: pod
          finalized: 14
         ------
     data:
       pointer: 000000000041FF08
       reference: 0000000000000000 (embedded in array memory)
    ------

In this case the nd::array's reference is still NULL, indicating its
memory is combined with the nd::array, but the string data itself
is in a separate memory block.
