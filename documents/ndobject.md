The DyND NDObject
=================

The DyND ndobject is a multidimensional data storage container, inspired
by the NumPy ndarray and based on the Blaze datashape system. Like NumPy,
it supports strided multidimensional arrays of data with a uniform
data type, but has the ability to store ragged arrays and data types
with variable-sized data.

NDObject Structure
------------------

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

In the DyND ndobject's Python exposure, the same data is:

    >>> a = nd.ndobject([[1,2,3],[4,5,6]])
    >>> print(a.debug_repr())
    ------ ndobject
     address: 00000000062090F0
     refcount: 1
     dtype:
      pointer: 0000000006209060
      type: strided_dim<strided_dim<int32>>
     metadata:
      flags: 3 (read_access write_access )
      dtype-specific metadata:
       strided_dim metadata
        stride: 12
        size: 2
        strided_dim metadata
         stride: 4
         size: 3
     data:
       pointer: 0000000006209140
       reference: 0000000000000000 (embedded in ndobject memory)
    ------

Some things that were metadata in NumPy ndarrays have become part of
the dtype in DyND ndobjects. The fact that this is a two-dimensional
strided array is encoded in the DyND dtype. One way to think of this
dtype is that it is a strided array of strided int32 arrays.

In the debug printout of the ndobject, there is first some metadata
about the ndobject, currently only access permission flags. The
dtype-specific metadata is determined by the structure of the dtype.
In this case, each strided_array part of the dtype owns some metadata
memory which contains its stride and dimension size. The int32 doesn't
require any additional metadata.

The data consists of a pointer to the memory containing the array elements,
and a reference to a memory block which holds the data
for the ndobject. In this case, the data has been embedded within
the same memory allocation as the ndobject metadata.

Memory Blocks
-------------

Data for ndobjects is stored in memory blocks, which are low level
reference counted objects containing allocated memory or a
handle/reference from some other system. For example, when a NumPy ndarray
is converted into an ndobject, a PyObject pointer and a version of
Py_DECREF which grabs the GIL is held by the memory block.

The memory block itself doesn't know where within it the data an
ndobject needs is, so whereever ndobjects need memory block references,
they also need a raw data pointer.

### Indexing Example

Here's a small example showing the result of a simple
indexing operation.

    >>> a = nd.ndobject([1,2,3])
    >>> print(a.debug_repr())
    ------ ndobject
     address: 000000000415C730
     refcount: 1
     dtype:
      pointer: 0000000006209160
      type: strided_dim<int32>
     metadata:
      flags: 3 (read_access write_access )
      dtype-specific metadata:
       strided_dim metadata
        stride: 4
        size: 3
     data:
       pointer: 000000000415C770
       reference: 0000000000000000 (embedded in ndobject memory)
    ------

    >>> print(a[1].debug_repr())
    ------ ndobject
     address: 000000000415AA10
     refcount: 1
     dtype:
      pointer: 0000000000000004
      type: int32
     metadata:
      flags: 3 (read_access write_access )
     data:
       pointer: 000000000415C774
       reference: 000000000415C730
        ------ memory_block at 000000000415C730
         reference count: 2
         type: ndobject
         dtype: strided_dim<int32>
        ------
    ------

In the printout of a[1], the first thing to note is the
dtype pointer, it's just the value 4. This is because
for a small number of builtin dtypes, their dtype representation
in the ndobject is just a type id.

Compare the pointer and reference of a[1] with that of a.
The pointer is 4 greater, as expected for indexing element 1 with
a stride of 4. The reference is the same as the ndobject a's
address, which you can see at the top of the printout. That's because
the array data was embedded in the ndobject's memory, so a reference
to that ndobject gets substituted for NULL while indexing.

### NumPy Example

Here's an example of an array sourced from NumPy. To make it
more interesting, we cause the memory of the array to be unaligned.

    >>> mem = np.zeros(9, dtype='i1')
    >>> a = mem[1:].view(dtype='i4')
    >>> b = nd.ndobject(a)
    >>> print(b.debug_repr())
    ------ ndobject
     address: 0000000006208500
     refcount: 1
     dtype:
      pointer: 0000000006208FD0
      type: strided_dim<unaligned<int32>>
     metadata:
      flags: 3 (read_access write_access )
      dtype-specific metadata:
       strided_dim metadata
        stride: 4
        size: 2
     data:
       pointer: 0000000003699541
       reference: 00000000041586B0
        ------ memory_block at 00000000041586B0
         reference count: 1
         type: external
         object void pointer: 000000000479B450
         free function: 000007FEEC181988
        ------
    ------

Because the data isn't aligned, the ndobject can't have a straight
int32 dtype. The solution is to make an unaligned<int32>
dtype, which has alignment 1 instead of alignment 4 like int32.

The data reference is an external memory block here. The "object void pointer"
is a pointer to the PyObject*, and the "free function" is a function which wraps
Py_DECREF in some code to ensure the GIL is being held. Unfortunately, this means
that freeing this object will be more expensive than normal, but there isn't really
another option that permits ndobjects to be used safely across multiple threads.
For memory blocks themselves, an atomic increment/decrement is used to provide this
thread safety.

### Variable-Length String Example

The default string dtype for dynd is parameterized by its encoding and is
variable-length.  This is handled by having a memory block reference in the
string's metadata, and the primary string data itself being a pair of pointers
to the beginning and one past the end of the string. For an example, here's
how a single Python string converts to an ndobject:

    >>> a = nd.ndobject("This is a string")
    >>> print(a.debug_repr())
    ------ ndobject
     address: 0000000006208550
     refcount: 1
     dtype:
      pointer: 000000000415AA50
      type: string<'ascii'>
     metadata:
      flags: 5 (read_access immutable )
      dtype-specific metadata:
       string metadata
        ------ memory_block at 00000000041586D0
         reference count: 1
         type: external
         object void pointer: 000000000494F3E8
         free function: 000007FEEC181988
        ------
     data:
       pointer: 0000000006208588
       reference: 0000000000000000 (embedded in ndobject memory)
    ------

What you will notice here is that the ndobject is pointing directly at the
Python string data, and has been flagged as immutable, just like Python
strings are. For comparison, let's do an array of unicode strings (these
examples are in Python 2.7).

    >>> a = nd.ndobject([u"This", u"is", u"unicode."])
    >>> print(a.debug_repr())
    ------ ndobject
     address: 00000000062090F0
     refcount: 1
     dtype:
      pointer: 0000000006209060
      type: strided_dim<string<'ucs-2'>>
     metadata:
      flags: 3 (read_access write_access )
      dtype-specific metadata:
       strided_dim metadata
        stride: 16
        size: 3
        string metadata
         ------ memory_block at 00000000062085A0
          reference count: 1
          type: pod
          finalized: 28
         ------
     data:
       pointer: 0000000006209138
       reference: 0000000000000000 (embedded in ndobject memory)
    ------

In this case it made a copy of the strings, into a POD (plain old data)
buffer instead of pointing at the original string.

