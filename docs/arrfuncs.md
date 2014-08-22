# DyND ArrFuncs

The ArrFunc, short for Array Function, is DyND's abstraction
to represent computation on ``nd::array`` values. It bundles
together a signature, some arbitrary data, and functions
for resolving types and instantiating a ckernel to
execute.

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
signatures, which encompass both dimensions and data types 

Some problems with this system include:

* Parameterized types like the datetime64 were added
  much later than ufuncs, and do not fit with the simple
  type id approach used to define overload signatures.
* The definition of overload resolution is somewhat ad
  hoc. It's well defined for the list of built-in type
  signatures, but not for interactions between different
  plugin types in the dictionary of extension signatures.


The first ``ufunc`` feature we'll look at is the ability
to provide a number of overloads, based on type ID.
In the ``gufunc`` this is extended with a "core dimensions"
signature that specifies dimensions to process all at once
instead of via element-wise broadcasting rules.

Some drawbacks in how this works include:

* Basic type resolution is implemented via a linear list
  of type id signatures. The datetime64 parameterized type
  doesn't really fit in well here.
* For pluggable types, there is an additional dictionary
  of signatures, which gets looked at after the linear list
  is exhausted.
* Only one core dimension signature is allowed, so it
  doesn't work for overloading something like gradient
  via something like ``(M * N * T) -> M * N * 2 * T``
  and ``(M * N * R * T) -> M * N * R * 3 * T``.

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
