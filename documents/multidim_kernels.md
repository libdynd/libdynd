Multi-dimensional DyND Kernels
==============================

The [DyND Kernel Documentation](kernels.md) describes
the basic memory format of DyND kernels, and how they
are constructed with kernel factories such as
'make_assignment_kernel' and 'make_comparison_kernel'.

This document goes into some more depth into how the
hierarchical kernel structure typically mirrors the
hierarchical structure of the dtypes involved.

Broadcasting Array Dimensions
-----------------------------

Broadcasting is an idea used in NumPy and other array
programming systems to flexibly and predictably combine
arrays of different shapes/dimensions. In Blaze, broadcasting
is a part of the type unification process.

The basic idea is that when you have one value, you
can "broadcast" it across many values. The simplest,
and most widely used version of this idea is
broadcasting scalars to array types. If M is a matrix,
then the 3 in 3*M is broadcast to all the elements of M.

In NumPy and DyND, this is generalized
in two ways.

### Broadcasting to New Dimensions

If two input arrays have a different number of dimensions,
the array with fewer dimensions is broadcast across the
leading dimensions of the other. For example, if A has
shape (2, 3), and B has shape (5, 2, 3), line up the
shapes in a right-justified manner to see how it is
broadcast:

```
   (2, 3) # A
(5, 2, 3) # B
---------
(5, 2, 3) # Result
```

This is the rule that scalars fall under as well,
for example if A is a scalar, it has zero dimensions,
so the equation becomes:

```
       () # A
(5, 2, 3) # B
---------
(5, 2, 3) # Result
```

### Broadcasting Dimensions Together

If matched up dimensions of two input arrays are different,
and one of them has size 1, it is broadcast to match the
size of the other. Let's say B has the shape (5, 2, 1) in the
previous example, so the broadcasting happens as follows:

```
   (2, 3) # A
(5, 2, 1) # B
---------
(5, 2, 3) # Result
```

### Generalizing to Ragged Arrays

For ragged arrays, where one of the dimensions may have
different sizes depending on the index in preceding arrays,
this can be easily generalized. The broadcasting is best done
at evaluation time, to avoid requiring multiple passes through
the array data.

Consider the two arrays A = [[1], [2, 3]] and B = [[4], [5]].
Their shapes are respectively (2, VarDim) and (2, 2), so
the broadcasting occurs as:

```
(2, VarDim) # A
(2,      1) # B
-----------
(2, VarDim) # Result
```

Now consider B = [[4, 5], [6, 7]]. Its shape is (2, 2), so
the broadcasting now occurs as:

```
(2, VarDim) # A
(2,      2) # B
-----------
(2,      2) # Result
```

Assignment Kernels That Broadcast
---------------------------------

Assignment kernels are the simplest case of
broadcasting in kernels. In an assignment, the
input is allowed to broadcast to the output,
but not vice versa.

### Scalar To One-Dimensional Array Example

Let's start with an example
that broadcasts a scalar to a one-dimensional
strided array.

```python
>>> from dynd import nd, ndt
>>> a = nd.ndobject([1,2,3])
>>> b = nd.ndobject(4)
>>> a.dtype
nd.dtype('strided_dim<int32>')
>>> b.dtype
ndt.int32
>>> a[...] = b
>>> a
nd.ndobject([4, 4, 4], strided_dim<int32>)
```

The broadcasting for this, following the notation
of our previous examples, looks like:

```
 () # Input
---
(3) # Output
```

Let's trace through how DyND creates the kernel that
performs this broadcasting assignment.

...

Making Efficient Strided Kernels
--------------------------------

One of the problems with the hierarchical kernel definitions presented
so far is that they always process array data in C order. For arrays
which are in F order or whose axes are permuted arbitrarily, it would
be better to process the axes in a different order. Additionally,
for elementwise operations on simple strided data, it is often possible
to coalesce the dimensions together and end up with a single strided
loop.

The design for doing this using the 'kernel_request_t' parameter to
the make_assignment_kernel functions. There are two kernel requests
for assignment kernels, 'kernel_request_single' and 'kernel_request_strided'.
These create kernels with a function prototype for assigning a single value,
and for assigning a strided array of values, respectively. Two new
kernel requests are added for simple strided dimensions,
'kernel_request_single_multistride' and
'kernel_request_strided_multistride'.

The idea of the 'multistride' kernel requests is that they accumulate
the dimension shape and strides in the kernel data as they go, without
actually creating the kernel until they reach a dtype that isn't
a simple strided dimension. That kernel factory calls a function to
turn the list of sizes and strides into a kernel, then adds itself
as a child.

The function which handles converting the 'multistride'
request into a kernel can analyze the shape and strides, for example
reordering them, coalescing them, and converting them into a blocked
assignment if the source and destination have incompatible striding
patterns. In the case of builtin assignment, we could add special
case kernels for two or three dimensional assignment to further
optimize the common cases.

