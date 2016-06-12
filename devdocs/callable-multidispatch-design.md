# Multiple Dispatch for DyND

We are on a quest to make DyND callables highly general and dynamic, but still perform well with low overhead. A
multiple dispatch implementation appears to be the best way to do this generally. To build a multiple dispatch
system, we need to define the semantics, i.e. the abstract model specifying how a set of input arguments chooses
between a set of signatures, and have a clear idea of how code implementing those semantics can be fast.

This fits into the DyND system as one of the more user-facing components, where the set of users includes programmers
of C++, Python, or other binding languages, and potentially also technically-minded users of node-based computation
flow graph editors. In this role, it's bridging a larger abstraction gap than many other parts of DyND, and
poses a greater challenge to get right.

## The `Matches` Partial Ordering

DyND has a pattern matching mechanism, where types cannot be explicitly instantiated in memory, but instead form
patterns which correspond to sets of concrete types. Defining the pattern matching in terms of (potentially
infinite) sets of concrete types means we can define a partial ordering of all DyND types in terms of the
subset relation.

The main thing that this misses from NumPy's ufuncs or C++ and Java function overloads is implicit promotion
of types. This is a little different than `matches`, because passing an `int16` to a function accepting `int32`
can't be done directly, it requires inserting type conversion code to fix up. It's unclear whether it would
be worth working out how type promotion could work in the context of multiple dispatch with the
`matches` partial ordering, or some other approach based on the `Kind` pattern types is preferable. Postponing
the resolution of this until the multiple dispatch is functioning well seems reasonable.

## An Efficient Implementation

An efficient implementation of DyND's main pluggable dispatch mechanism is crucial. The main candidate for
this is a decision tree based on the type IDs and potentially other properties of the input arguments. Let's
start with some examples to see how this might work.

Let's start with an operation with a few concrete signatures.

```
A: (int8, int8) -> int8
B: (int16, int16) -> int16
C: (float32, float32) -> float32
```

The three signatures are not ordered by `matches`, so we have three disconnected nodes `A`, `B`, `C`,
and any of the 6 possible orderings are topologically sorted. In this particular case, a hash table
whose key consists of the type id pairs will get to the correct signature in one step.

Let's do a gradient function.

A: (N * float32) -> N * float32
B: (M * N * float32) -> M * N * 2 * float32
C: (Fixed ** N * float32) -> Fixed ** N * N * float32

Here, `A` and `B` are not ordered, but `C` is ordered after both of them. With only one argument, and
assuming the default constraint on a dimension type variable is that it be `Fixed`, our decision tree
looks like this:

```
fixed -> (potential matches {A, B, C})
|    \                 \
|     float32 -> !{A}   \
|                        <other> -> !{}
|
fixed -> {B, C}
|    \                 \
|     float32 -> !{B}   \
|                        <other> -> !{}
|
fixed -> {C}
|    \
|     float32 -> !{C}
|
fixed -> {C}
|    \
|     float32 -> !{C}
|
... (repeating forever)
```

What this reminds me of is the subset construction of a DFA from an NFA, and I think we can use exactly
this to construct a matching DFA. The key part would be in deciding what the outgoing edges of a DFA
node should be in doing the analogous subset construction.

One option is for an outgoing edge to be always based on the type id of one of the arguments.
Suppose we are at a DFA node representing the signature set {A, B, C, D, ...}. We go through each
argument, construct the resulting DFA nodes one would get to, then use some criterion like choosing the
one whose destination DFA nodes have the smallest signature sets to decide. Then, repeat for each of the
seen but unprocessed DFA nodes until there are none left.

