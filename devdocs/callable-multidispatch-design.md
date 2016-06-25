# Multiple Dispatch for DyND

We are on a quest to make DyND callables highly general and dynamic, but still perform well with low overhead. A
multiple dispatch implementation appears to be the best way to do this generally. To build a multiple dispatch
system, we need to define the semantics we want and have a clear idea of how code implementing those semantics
can be fast.

This is one of the more user-facing components of DyND, as the creation of multiple signatures to match will be
created in every binding that includes a way to specify computation, both programming languages like C++ and
Python as well as node-based flow graph editors. In this role, it bridges a larger abstraction gap than many
other parts of DyND, and poses a greater challenge to get right.

## The `Matches` Partial Ordering

DyND's type system includes a pattern matching mechanism, based on types which cannot be explicitly instantiated
in memory, but instead form patterns which correspond to sets of concrete types. Because the pattern matching
mechanism effectively defines (potentially infinite) sets of concrete types, there is a natural partial ordering
of all DyND types in terms of the subset relation.

The main thing that this misses from NumPy's ufuncs or C++ and Java function overloads is implicit promotion
of types. This is a little different than `matches`, because passing an `int16` to a function accepting `int32`
can't be done directly, it requires inserting type conversion code to fix things up. It's unclear whether it would
be worth working out how type promotion could work in the context of multiple dispatch in concert with the
`matches` partial ordering, or some other approach based on the `Kind` pattern types is preferable. Postponing
the resolution of this design until the `match`-based multiple dispatch is functioning well seems reasonable.

## Definition of the Multiple Dispatch Resolution

A multiple dispatch callable consists of a set of callables, which may have concrete or symbolic signature
types `S_all = {S0, S1, ..., Sn}`. When making a call, DyND has a list of input argument types, which we will
bundle together as a tuple type `A` for the purpose of this discussion. The task is to find the best match
for `A` among those signatures. DyND callable signatures include both positional and keyword parameters, and
this matching only applies to the positional. To simplify the discussion, let's assume there are no
keyword parameters.

The a set `M` of resolved signatures is defined as follows. Specifically, the `matches` partial ordering
allows us to define whether one match is more specific than another, and we use that to limit `M`.

```
# All the signatures which match the input arguments
M_candidate = {S in S_all | T matches S}
# Only the most specific ones, i.e. signatures which don't have a more specific one in the set
M = {S in M_candidate | for all S' in M_candidate, not (S' matches S and not S matches S') }
```

This set `M` could be empty, have size 1, or be arbitrarily large. In the case where its cardinality
is greater than 1, we may want to raise an error which describes the ambiguity and how one might
eliminate it.

## Multiple Dispatch Via Topological Sort

You can see an example implementation of multiple dispatch implemented in terms of topological sort
in the Python multidispatch library, at https://github.com/mrocklin/multipledispatch/blob/master/multipledispatch/conflict.py#L56.

With the signatures topologically sorted according to `match`, the first matching signature encountered by
iterating forward from the beginning will be an element of `M`. With some additional bookkeeping, we can also
track how much farther in the list to look in order to find the full set and determine whether there's an
ambiguity or not.

This linear search through a topologically sorted list, in a hand-wavy way, is approximately what NumPy
does in ufunc matching. There, the signatures are restricted to scalar functions, and the topological sort
is based on whether the data conversion of the argument to the signature types is "safe". No actual topological
sort is done, rather when one writes a ufunc, the list of signatures that gets put in explicitly needs
to be in this order for things to make sense.

For small sets of signatures and simple integer type ids, this works well, but it would be nicer to restrict
the set of potential matches without doing full-blown match checks against a large number of signatures. We
need an algorithm which will zero in on the candidate signatures quickly.

## Implementing Faster Dispatch as an Explicit Decision Tree

What is described here is inspired by the
[powerset or subset construction](https://en.wikipedia.org/wiki/Powerset_construction),
a technique for constructing a DFA (Deterministic Finite Automaton) from an NFA
(Non-deterministic Finite Automaton). We are using an analogous construction from the power set of `S_all`.
A guiding principle is that it's ok to do quite expensive things during construction of the decision tree,
but only very cheap things while using it to do the match. Our algorithm will model the execution of
the decision tree by successively peeling away layers of the signature types as components of them are
matched, keeping track of these partially-matched signatures for the set of all signatures that are
still candidates.

### How the Decision Tree Works

The input to the decision tree is an array of `m` types, `TInp = [T0, T1, ..., Tm]`. The first decision criterion
is `m`, restricting to the subset of signatures with `m` positional arguments. This gives us the subset
`S_working` of `S_all` as the starting point for the meat of the algorithm, which works recursively.

In addition to the array of `m` input types, which will remain untouched, the decision tree will track
a vector `TWork` of borrowed type references (stored as `ndt::type *` or potentially the more efficient
`base_type *`). The original array has ownership of the types, so there's no reason to aquire any
references of type subcomponents.

Similarly, a vector `SWork` of strings and `IWork` of integers are kept to track strings such as field
names and integers such as dimension sizes. These operate in the same fundamental way as `TWork`, so
the discussion will be primarily about it.

A node in the decision tree branches based on one of several possible criteria. The most common one
is a table or hash-map lookup using the type id of one entry in `TInp` or `TWork`, or the value of
one entry in `SWork` or `IWork`. Another one is equality comparison of a value in `TInp`, `TWork`,
`SWork`, or `IWork` against a constant value. The final one is equality comparison between two type
ids in either `TInp` or `TWork`, between two strings in `SWork`, or between two integers in `IWork`.
Think of the node as a little snippet of code that looks at some part of the current working data,
and uses it to decide between two or more child nodes.

To motivate these three decision operations, let's look at simple example signature sets that can
be distinguished between them. The first is most obvious, a set like
`{(int8) -> void, (int16) -> void, (float32) -> void}` is immediately decided by a table lookup
of `T0`. For the type id equality, consider the set `{(int8) -> void, (Scalar) -> void}`. Technically the
table lookup would work, but just an equality check against the `int8` type id is simpler.
For integer equality, consider `{(3 * Scalar) -> void, (Dim * Scalar) -> void}`. For string equality,
`{({x: int8, y: int8}) -> void, ({x: int8, r: int}) -> void}`.

Each decision either shrinks the current `S_working`, or exposes new information that can later
shrink it. For that last struct example, the decision tree will start with "Does TInp[0] have struct type id?",
where "no" branches to return no valid match, and "yes" triggers an action that explodes the struct
into the working vectors (one integer, the number of struct fields, in `IWork`, all the field types
in `TWork`, and all the field names in `SWork`. The next node asks "Is IWork[0] 2?" where "no"
branches to return no valid match, and "yes" continues straight to a node which asks "Is SWork[1] 'y'".
If there was a third signature that had yet a different name for the second field, this last decision
would be a string hash-map lookup instead of a string equality check.

In that example, each branch of the final decision has only one candidate signature, which leaves us with
a design choice. Do we stop the decision tree here, and use `matches` to confirm or deny that single
candidate, or do we make a bigger decision tree which finishes the job definitively? The former is probably
simpler, but the latter might be more efficient, especially when we JIT-compile the decision tree at
some point in the future.

### How We Build the Decision Tree




### Example 1

Let's start with an operation with a few concrete signatures.

```
A: (int8, int8) -> int8
B: (int16, int16) -> int16
C: (float32, float32) -> float32
```


### Example 2

Let's do a gradient function.

A: (N * float32) -> N * float32
B: (M * N * float32) -> M * N * 2 * float32
C: (Fixed ** N * float32) -> Fixed ** N * N * float32

