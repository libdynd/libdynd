# DyND ArrFuncs

The ArrFunc, short for Array Function, is DyND's abstraction
to represent computation on ``nd::array`` values. It bundles
together a signature, some arbitrary data, and functions
for resolving types and instantiating a ckernel to
execute.

The goal for arrfuncs is to represent all array functions
within dynd, so that vtables for dynd types consist primarily
of arrfunc objects. The trade-offs in designing it are
mostly between making things static so they can be repeated
efficiently over many array elements, and making things
dynamic and flexible.

## Motivation via NumPy UFunc Features

* http://docs.scipy.org/doc/numpy/reference/c-api.ufunc.html
* http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html

NumPy has ``ufunc`` and ``gufunc`` abstractions which
play a similar role. We'll use them to motivate the
``nd::arrfunc`` design. These objects package together
the following data/functionality:

* A single function prototype for implementing kernels as
  one-dimensional loops of the function
  (``PyUFuncGenericFunction``).
* A list of overloaded kernels for built-in types, and
  a dictionary of overloaded kernels for pluggable types.
* The ``gufunc`` adds a "core dimensions" signature to
  be able to provide inner dimensions to the kernel.
* Broadcasting of additional dimensions, so operations
  are always element-wise.

The nd::arrfunc abstraction is designed to break these
apart into separate composable features, all working through
the same interface.

### Function Signatures

Types for overloads are defined via datashape type
signatures. For simple arrfuncs, these signatures contain
concrete types, like ``(bool, bool) -> bool``. Because
dimensions can be specified as part of datashapes, the
``gufunc`` core dimensions can be part of a signature like
``(M * bool, M * bool) -> bool``.

### Overloading

* https://github.com/libdynd/libdynd/issues/97

The concept for overloading is that an overloaded arrfunc
contains an array of arrfuncs with simple signatures,
and exposes a function signature which matches against
input signatures for any of the overloads.

## Use Cases

### Summary of take-aways from the below

* Might want to have separate *array parameters* and
  *dynamic parameters*, perhaps indicated via positional
  vs keyword arguments. I believe Julia does something
  along these lines, with positional parameters partaking
  in the multiple dispatch, and keyword parameters not.
* In-place operation ``(strided * T) -> void`` versus
  returning a modified result ``(strided * T) -> strided * T``,
  for example a ``sort`` operation, would require readwrite
  parameters in ckernels.
* An operation may support returning views into an input
  array rather than making copies of data. Need to be able
  to specify that, and properly propagate immutability or
  readonlyness.
* A way to construct the output type and output arrmeta
  as part of instantiation. This is needed to support
  what the [] indexing does, and is probably needed in
  general to support handling ownership references when
  returning views.

### Simple Reduction: Sum

For ``sum``, we have one array parameter and a few extra
parameters which specify how the sum is applied (``axis``
and ``keepdims``). Let's classify the latter as dynamic
parameters.

To determine the type of the result, we need to know
both the type of the array and the values of the dynamic
parameters. For example
``sum([[3 * 5 * int32]], axis=1, keepdims=True)`` could
have output type ``3 * int32``.

### Sort

In NumPy, ``sort`` takes a few keyword arguments, including
``axis``, ``kind`` (quicksort, etc), and ``order`` (a
special option for structured arrays indicating field order).

I suspect ``kind`` would be better served via multiple
sort functions (e.g. std::sort, std::stable_sort,
std::partial_sort in C++ STL). The ``order`` parameter is
potentially confusing, as it could be expected to be
a way to reverse the sort order for example.

Python's sort function has two keyword parameters,
``key=`` and ``reverse=``. For DyND, emulating these might
be a better option than the numpy ``order=``.

NumPy interprets the ``axis`` argument to mean sort each
1D sliced array along that axis independently. Another
intuitive behavior might be to sort at that axis, sorting
later axes lexicographically. This would be more intuitive
for a ragged array, or an array with labels along these
later axes, for example.

Python offers two variants of sorting - an in-place method
on lists, and a ``sorted`` function which returns a sorted
version of its argument. DyND should support both as well.

### String Split

With a signature like ``(string) -> var * string``, with
a keyword argument ``sep=``, this operation could return
views into data of the original string (though would be
allocating memory for the variable dimension). Whether
this is possible will depend on the type and arrmeta of the
input array.

For example, one could imagine memory-mapping a text file,
splitting the file into lines, then splitting each line
into fields, resulting in a ``var * var * string`` array
whose string data is still pointing into the original
memory-map.

At the ckernel level, one can see whether a view should
be created by looking at the arrmeta of the input and
output strings. If the data reference is the same, then
the output should be pointing into the same memory.

At the arrfunc level, whether a view is supported may
be something to specify via flags somewhere. It would probably
be a request during the instantiation step, with three
levels following the ``nd.array``, ``nd.asarray``, ``nd.view``
logic.

### Reductions with Extra State (Kahan Summation)

Kahan summations requires tracking both a sum and a
running compensation. We can represent this in
three pieces:

1. An initialization value [0, 0] with type
   ``(float64, float64)``.
2. A reduction kernel ``(float64) -> (float64, float64)``
3. A finishing kernel ``((float64, float64)) -> float64``

For supporting parallelism, a fourth piece:

4. A combining kernel ``((float64, float64)) -> (float64, float64)``.

Would it be worth having a conventional way to pack together
these components?

### Indexing (operator[])

Indexing is presently handled as a special case with a pair
of virtual functions on the types. It would be nice if
indexing could be represented as an arrfunc too. Early on
in dynd, having indexing always produce a view was set out
as a desirable property. If there were a way to specify
view/not-view as for the string splitting use case, it
would apply equally well here.

Indexing requires two steps:

1. Matching the index against the input type, producing
   the output type. Code calling the indexing operation
   can then allocate arrmeta for the output.

2. Matching the index and input type/arrmeta, filling
   the output arrmeta.

To be able to support this, the arrmeta instantiation
needs a way to create and populate the output arrmeta,
instead of relying on default creation with a possible
shape, as is done now.

### Range

DyND has ``nd.range``, which would ideally be modeled as
an arrfunc as well. It has a few signatures in its Python
binding:

```
nd.range(stop, dtype=None)
nd.range(start, stop, dtype=None)
nd.range(start, stop, step, dtype=None)
```

In the first signature, both ``stop`` and ``dtype`` affect
the type or arrmeta of the result. ``nd.range(5)`` produces
a size 5 ``strided`` array, while ``nd.range(10)`` produces
size 10. Same for ``start`` and ``stop``. As a consequence,
all parameters must be dynamic, and there are no array
parameters.

A possible signature for ``nd.range`` is
``(start: ?T, stop: T, step: ?T, dtype: ?type) -> strided * R``,
which will lower into a ckernel with signature
``() -> strided * R``.

### Reshape

A very common operation in quick test NumPy code is to create a
multi-dimensional array with increasing values as follows:

```
In [47]: np.arange(24).reshape(2, 3, 4)
Out[47]: 
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
```

There is a corner case of the ``reshape``
operation in NumPy that seem wise to restrict, namely if
the input is multi-dimensional, it gets implicitly flattened
in C-order before being reshaped. I think treating it always
as a 1D to ND operation is better,
something like ``(strided * T) -> Dims... * T``.

The NumPy ``reshape`` signature doesn't quite fit with the
idea of having positional arguments always be array parameters
and keyword arguments always be dynamic parameters, because
the shape needs to be variadic. If we accept the shape as
a single argument, it would look like
``(strided * T, shape: strided * intptr) -> Dims... * T``.

Python code would look like

```
>>> nd.reshape(nd.range(24), (2, 3, 4))
```

This is another function which would support viewing the
input data.

### Gradient/Hessian

Gradient and Hessian are natural numeric operations
to represent on a grid. With an extension to datashape
to allow constrained typevars as ``(A: <constraint>)``,
one can represent their signatures as

```
((Dims: fixed**N) * T) -> Dims... * N * T
((Dims: fixed**N) * T) -> Dims... * N * N * T
```

It might also be natural for these functions to have
an additional dynamic parameter specifying the method
of approximation.

## Datashape Function Signature Limitations

Datashape function signatures as presently defined don't
cover all the cases one might like.

### Pattern Matching Quirks

There are limitations to pattern matching as presently
designed which limit the expressivity one might desire.

1. No simple "match anything" type variable.
   A single type variable can only match against
   one dimension type or one data type. To match an
   arbitrary datashape requires a combination of an
   ellipsis typevar and a dtype typevar, like ``D... * T``.
2. Expressing a matrix multiplication signature
   analogously to the ``gufunc`` core dimensions as
   ``(M * N * T, N * R * T) -> M * R * T`` doesn't precisely
   capture the desired meaning, because the dimension
   typevars match against any dimension type rather than
   constraining the size of the dimensions they see.
   This means ``(strided * var * real, var * 3 * T) -> strided * 3 * real``
   is a valid match.
3. No mechanism defined to match against string or integer values
   in the signature, like ``(datetime[tz=TZ], datetime[tz=TZ]) -> units[int64, 'microseconds']``
   or ``({F0: S, F1: T}) : {F1: T, F0: S}``. The convention
   for typevars to begin with uppercase doesn't apply to the
   field names, so "F0" and "F1" aren't typevars but rather
   the literal field names, and a different syntax would be
   required to match the name here.

There are further cases where the output type
can't be easily represented with simple pattern matching,
like a reduction with ``axis`` and ``keepdim`` arguments.
See the [Elementwise Reduction document](elwise-reduction-ufuncs.md)
for details on this example. The dependent type mechanism
described there would probably be a little bit nicer with
a simple way to pattern match a whole datashape too.

### No Keyword Arguments

Python is a source of much inspiration for DyND, and as
such it would be natural for arrfuncs to support keyword
arguments. This would require an extension to datashape,
maybe to allow datashapes like
``(string, real, repeat: int, debug: string) -> int``.

In Python 3, keyword-only arguments were added, which might
be nice here too, however the syntax to choose isn't
immediately obvious.

### No Optional Arguments

This is particularly useful for keyword arguments, and
could be done simply using the option type in datashape.
The above example keyword argument example could be
``(string, ?real, repeat: ?int, debug: ?string) -> int``
to denote that the last three arguments are optional.

### No Variadic Arguments

The most general Python function signature is
``def fn(*args, **kwargs)``. Something to specify
a general function signature and variadic arguments
would be useful.
