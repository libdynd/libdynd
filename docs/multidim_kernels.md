Multi-dimensional DyND Kernels
==============================

The [CKernel Documentation](ckernels.md) describes
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
Their shapes are respectively (2, VarDim) and (2, 1), so
the broadcasting occurs as:

```
(2, var) # A
(2,   1) # B
--------
(2, var) # Result
```

Now consider B = [[4, 5], [6, 7]]. Its shape is (2, 2), so
the broadcasting now occurs as:

```
(2, var) # A
(2,   2) # B
--------
(2,   2) # Result
```

Assignment Kernels That Broadcast
---------------------------------

Assignment kernels are the simplest case of
broadcasting in kernels. In an assignment, the
input is allowed to broadcast to the output,
but not vice versa.

### Scalar To One-Dimensional Array Example

Let's start with an example that broadcasts a
scalar to a one-dimensional strided array.

```python
>>> from dynd import nd, ndt
>>> a = nd.array([1,2,3], access='rw')
>>> b = nd.array(4)
>>> nd.type_of(a)
ndt.type('strided * int32')
>>> nd.type_of(b)
ndt.int32
>>> a[...] = b
>>> a
nd.array([4, 4, 4], type="strided * int32")
```

The broadcasting for this, following the notation
of our previous examples, looks like:

```
 () # Input
---
(3) # Output
```

Let's trace through how DyND creates the kernel that
performs this broadcasting assignment. The function called
to create any assignment kernel is in called make_assignment_kernel
in 'dynd/kernels/assignment_kernels.hpp'. In this case, it will
call

```cpp
dst_dt.extended()->make_assignment_kernel(...);
```

The destination dtype is 'strided_dim', which is defined in
'dynd/dtypes/strided_dim_dtype.hpp'. This function has
a test comparing the the number of uniform dimensions of
the source and destination types.

```cpp
    // in this example, is "if (0 < 1)"
    if (src_dt.get_undim() < dst_dt.get_undim()) {
```

In this case, the src_stride of the strided assignment
kernel gets set to 0, and no dimensions from the source
dtype are peeled off. The next call is to make_assignment_kernel
with both the src and dst dtypes equal to 'int32', requesting
a strided kernel. The resulting kernel is precisely the
example in the [ckernel documentation](ckernels.md) with the
struct called strided_int32_copy_kernel_data, where the value
src_stride has been set to zero.

### Var To Strided Array Example

Our next example assigns from a ragged array to a strided
array, doing the broadcasting during the assignment.

```python
>>> from dynd import nd, ndt
>>> a = nd.array([[5, 6, 7], [8, 9, 10]], access='rw')
>>> b = nd.array([[1, 2, 3], [4]])
>>> nd.type_of(a)
ndt.type('strided * strided * int32')
>>> nd.type_of(b)
ndt.type('strided * var * int32')
>>> a[...] = b
>>> a
nd.array([[1, 2, 3], [4, 4, 4]], type="strided * strided * int32")
```

The assignment kernel here results in three
levels of kernel functions, a "strided to strided" function,
a "var to strided" function, and an "int32 to int32" function.

```cpp
    /** Typedef for a unary operation on a single element */
    typedef void (*unary_single_operation_t)(char *dst, const char *src,
                    ckernel_prefix *extra);
    /** Typedef for a unary operation on a strided segment of elements */
    typedef void (*unary_strided_operation_t)(
                    char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra);

    struct var_to_strided_copy_kernel_data {
        // The first ckernel_prefix ("strided to strided")
        unary_single_operation_t dim0_kernel_func;
        destructor_fn_t dim0_kernel_destructor;
        // Data for the first kernel
        intptr_t dim0_size;
        intptr_t dim0_dst_stride, dim0_src_stride;

        // The second ckernel_prefix ("var to strided")
        unary_strided_operation_t dim1_kernel_func;
        destructor_fn_t dim1_kernel_destructor;
        intptr_t dim1_dst_stride, dim1_dst_dim_size;
        const var_dim_dtype_arrmeta *dim1_src_md;

        // The final ckernel_prefix ("int32 to int32")
        unary_strided_operation_t scalar_kernel_func;
        destructor_fn_t scalar_kernel_destructor;
    };
```

The first kernel function, 'dim0_kernel_func', is exactly
the same as the one in the last example. The second kernel
function is defined in 'dynd/kernels/var_dimn_assignment_kernels.cpp',
as var_to_strided_assign_kernel. The destination is strided,
so the stride and size are stored for it. For the source, a
pointer to its arrmeta is copied, though the individual fields
needed could be copied as well to avoid the indirection.

The 'dim1_kernel_func' does a check for a broadcasting error
for each element it copies. If the size of the src dimension is
1 or equal to the size of the dst dimension, broadcasting or
copying can be done.

The sequence of calls which occur when
the kernel is called, assuming all single instead of
strided kernels, are as follows:

<table>
    <tr>
        <th>src data</th>
        <th>operation</th>
        <th>dst data</th>
    </tr>
    <tr>
        <td><b>[[1, 2, 3], [4]]</b></td>
        <td>strided to strided</td>
        <td><b>[[5, 6, 7], [8, 9, 10]]</b></td>
    </tr>
    <tr>
        <td>[<b>[1, 2, 3]</b>, [4]]</td>
        <td>&nbsp;&nbsp;&nbsp;var to strided</td>
        <td>[<b>[5, 6, 7]</b>, [8, 9, 10]]</td>
    </tr>
    <tr>
        <td>[[<b>1</b>, 2, 3], [4]]</td>
        <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;int32 to int32</td>
        <td>[[<b>5</b>, 6, 7], [8, 9, 10]]</td>
    </tr>
    <tr>
        <td>[[1, <b>2</b>, 3], [4]]</td>
        <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;int32 to int32</td>
        <td>[[1, <b>6</b>, 7], [8, 9, 10]]</td>
    </tr>
    <tr>
        <td>[[1, 2, <b>3</b>], [4]]</td>
        <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;int32 to int32</td>
        <td>[[1, 2, <b>7</b>], [8, 9, 10]]</td>
    </tr>
    <tr>
        <td>[[1, 2, 3], <b>[4]</b>]</td>
        <td>&nbsp;&nbsp;&nbsp;var to strided</td>
        <td>[[1, 2, 3], <b>[8, 9, 10]</b>]</td>
    </tr>
    <tr>
        <td>[[1, 2, 3], [<b>4</b>]]</td>
        <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;int32 to int32</td>
        <td>[[1, 2, 3], [<b>8</b>, 9, 10]]</td>
    </tr>
    <tr>
        <td>[[1, 2, 3], [<b>4</b>]]</td>
        <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;int32 to int32</td>
        <td>[[1, 2, 3], [4, <b>9</b>, 10]]</td>
    </tr>
    <tr>
        <td>[[1, 2, 3], [<b>4</b>]]</td>
        <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;int32 to int32</td>
        <td>[[1, 2, 3], [4, 4, <b>10</b>]]</td>
    </tr>
    <tr>
        <td>[[1, 2, 3], [4]]</td>
        <td></td>
        <td>[[1, 2, 3], [4, 4, 4]]</td>
    </tr>
</table>

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

