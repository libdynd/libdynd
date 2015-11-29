DyND NumPy-Like API
===================

This is a design document describing the philosophy
and details the DyND API for supporting
NumPy functionality. While NumPy is an amazing
inspiration for how the API should look, we are not
matching it precisely. Differences arise from factors
such as the datashape-based type system.

DyND API Design Guidelines
--------------------------

When designing DyND APIs, the following points
should guide the design:

 * An important target user is a domain expert who knows
   programming, not necessarily an expert programmer.
   Imagine how such a user might try to use DyND,
   and design the public-facing API so as to match their
   expectations or guide them to the right answer.
 * Another important target user is the expert
   programmer. Don't dumb down the API so that it
   tries to "do the right thing" without providing
   a well-specified abstraction that can be relied on.
   This is a major drawing point of Python.
 * When making an API choice, always match
   NumPy, Pandas, SciPy, etc unless a difference
   is needed for:
   * using the datashape type system.
   * supporting distributed or out of core arrays.
   * self-consistency or a clean API. This kind of
     decision should be agreed upon within the DyND
     team after a discussion.
 * Consider how a function generalizes from only
   fixed-size dimensions in NumPy to the `var`
   dimension type in DyND.
 * Consider using keyword-only arguments. These
   are supported in the Python 3 syntax, but can
   be emulated using **kwargs in Python 2 as well.
   For adding many optional arguments, they are
   good because they make user's code more
   self-documenting.

Array Creation
--------------

In NumPy, there is a suite of functions for creating
arrays in various ways, such as `numpy.ones`,
`numpy.empty`, and `numpy.arange`. The main difference
DyND has is that wherever a shape + dtype are
specified, a datashape needs to go instead.

Here are some of the NumPy array creation routines

 * `np.array` creates an array from data. NumPy
   has many related functions, like `np.asarray`,
   `np.asanyarray`, `np.ascontiguousarray`,
   `np.asmatrix`.
 * `np.empty` creates an array of uninitialized data.
 * `np.zeros`, `np.ones`, `np.full` create arrays
   initialized with a value.
 * `np.arange`, `np.linspace`, `np.logspace` create
   arrays with numerical ranges.
 * `np.eye`, `np.diag`, `np.tri`, etc provide
  linear algebra-oriented creations.

Array Reshape and Flatten
-------------------------

There is a common idiom in NumPy tutorials to
create test arrays as follows:

```python
>>> import numpy as np
>>> np.arange(6).reshape(2, 3)
array([[0, 1, 2],
       [3, 4, 5]])
```

One of the things we want to accomplish with DyND
is increase the predictability of certain kinds of
operations. NumPy's reshape tries to be smart,
taking a view when possible. Whether a view is
taken or a copy is made thus depends on the strides
of the array, and cannot be reliably predicted in
all cases.

The following is an example of this,
where `a` and `b` are views of the same data,
but reshaping them to flat arrays produces a copy
in one case but not the other.

```python
>>> a = np.arange(6).reshape(2, 3)
>>> a.ctypes.data
127557280

>>> b = a[:,:2]
>>> b.ctypes.data
127557280

>>> a.reshape(6).ctypes.data
127557280

>>> b.reshape(4).ctypes.data
127557344
```

In DyND, we would like both of these flattening
operations to behave "like a view". The way we
can do this for the latter is have the result
support iteration and indexing through appropriate
APIs or protocols.

Additionally, the case of reshaping a multidimensional
array to another multidimensional array has a hidden
flattening operation. Here's the kinds of things
you can get with just a two dimensional reshape:

```python
>>> np.arange(6).reshape(2,3).reshape(3,2)
array([[0, 1],
       [2, 3],
       [4, 5]])

>>> np.arange(6).reshape(2,3).reshape(3,2, order='F')
array([[0, 4],
       [3, 2],
       [1, 5]])

>>> np.arange(6).reshape(2,3,order='F').reshape(3,2, order='F')
array([[0, 3],
       [1, 4],
       [2, 5]])

>>> np.arange(6).reshape(2,3,order='F').reshape(3,2)
array([[0, 2],
       [4, 1],
       [3, 5]])
```

It seems likely that flatten and reshaping a one
dimensional array to a multidimensional array are
the only important cases, so we may consider
limiting reshape in DyND this way.

Elementwise UFuncs
------------------

NumPy's large library of ufuncs is an essential
feature of the system. DyND needs to provide the
same ufuncs, though with some different choices
in the general interface.

http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs

DyND arrfuncs encompass both NumPy ufuncs and
generalized ufuncs in one swoop, by allowing
the arguments to be typed with multi-dimensional
datashape types.

### Reductions Derived From UFuncs

NumPy automatically adds a few reduction methods
to all ufuncs. This means we can call
the nonsensical `np.sin.reduce([1,2,3])`, for
example, which raises an exception. It would be
better in DyND to simply not add these reductions
in such cases.

Not all array programming systems agree on the
definition for these reductions, for example
NumPy uses an order of operations from left to
right, while J and other APL dialects use right
to left.

```python
>>> np.subtract.reduce([1, 2, 3])
-4
```

NumPy calculated `(1 - 2) - 3`, whereas
J calculates `1 - (2 - 3)`.

```
   -/ 1 2 3
2
```

Some features we do want when deriving reductions
from binary ufuncs are whether certain properties are
satisfied by the operation. Even in cases where a
property is only approximate (e.g. associativity
in floating point addition, try 1e-16 + 1 + -1),
we will usually want to indicate it is so that
calculations may be reordered for efficiency.

A previous incarnation of DyND included some
development towards this, with per-kernel customization
of the associativity flag, commutativity flag, and
identity element.
https://github.com/libdynd/dynd-python/blob/master/doc/source/gfunc.rst#creating-elementwise-reduction-gfuncs

Elementwise Reductions
----------------------

Some reductions can be computed by visiting each
element exactly once, in sequence, using finite
state. The most common NumPy reduction
operations, `all`, `any`, `sum`, `product`,
`max`, and `min` all fit this pattern. Some
statistical functions like `mean`, `std`, and `var`
also fit this pattern, with slightly more sophisticated
choices for the accumulator state, as well as a finishing
operation.

NumPy has two keyword arguments that are worth
keeping in this kind of operation for DyND,
`axis=` and `keepdims=`. The parameter to `axis`
may be a single integer, or, when the operation
is commutative and associative, a tuple of integers.
The `keepdims` parameter keeps the reduced dimensions
around as size 1 instead of removing them, so the
result still broadcasts appropriately against
the original.

To see how `mean` fits into this pattern, consider
a reduction with datashape '{sum: float64; count: int64}',
identity `(0, 0)`, and function

```python
def reduce(out, value):
    out.sum += value
    out.count += 1

# alternatively, chunked
def reduce_chunk(out, chunk):
	out.sum += nd.sum(chunk)
	out.count += len(chunk)
```

A final division maps this to `float64`. When
defining this kind of kernel, it would also
be advantageous to provide a function to combine
two partial results, so as to allow DyND to
parallelize the operation

```python
def combine(out, out_temp1, out_temp2):
    out.sum = out_temp1.sum + out_temp2.sum
    out.count = out_temp1.sum + out_temp2.count

def combine_destructive(out, out_temp):
    out.sum += out_temp.sum
    out.count += out_temp.count
```

Additional Discussion
---------------------

Some additional points for discussion, raised by Peter:

 * Do we move a lot of the top-level `numpy` functions into
   methods, and try to stick to more of a "Only one way to do it"
   API, so that chaining of methods is natural like this?
 * What does everyone think about a mechanism for hinting?
   This would be very useful for parallelism and distributed
   cases, and can be built in either as kwargs for reduction,
   filter, and join/merge funcs, or as an explicit method that
   is called in the chain:

```
ary.ufunc1().ufunc2().reduce_or_filter(array2, shapehint=(N,...)).ufunc3()
```

 * Similarly, what about specifying chunksize as a hint, for
   when we use these DAGs to process streams? So we could
   change up the processing chunksizes at certain points?
