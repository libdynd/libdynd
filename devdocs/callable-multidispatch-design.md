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
# All the signatures which match the input arguments (where A is the input arguments)
# NOTE: X.matches(Y) means that Y is a match to the possibly symbolic type X
M_candidate = {S in S_all | S.matches(A)}
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

To build the decision tree, we explicitly track `S_working`, information about the state of all the working
vectors, a symbolic version of the working vectors for every signature in `S_working`, and a set of equations
that remain to be checked before we can declare a match. Each node in the decision tree is based on one of
these equations, and following an edge to a child node involves removing that equation, applying the knowledge
gained to the existing symbolic representations, and adding zero or more new equations. Let's refer to
the symbolic state of the input with the names `TInp`, `TWork`, etc, and the symbolic state of a candidate
signature `S` as `S.TInp`, `S.TWork`, etc.

After branching based on the number of arguments, the equations we have are `matches` between the
input argument types and the signature parameter types. They are of the form `S.TInp[i].matches(TInp[i])`.
In the case of a `matches` equation, we generate a candidate type id test, which selects an edge based on
`TInp[i].typeid`. For types fully identified with their type id, like `int32`, no new equations need to get
added, but for types with parameters, those parameters get placed into the work vectors, and new equations
in terms of those work vectors are added. These new equations are such that they and the decision tree node
are all satisfied if and only if the original matches equation was satisfied.

This is done for each signature in `S_working`, and for each candidate we produce a heuristic score.
The candidate with the best score is used to create the decision tree node, and then this whole procedure
is applied recursively with the new symbolic state and set of equations.

When matching against type variables in the signature, an additional symbol table may be tracked to
ensure consistency between all usages of the same variable. When a decision tree edge fully determines
the type variable's value, we simply substitute that value into all the symbolic state for that signature.
In all other cases, we append the working type on `TWork`, and track the type variable to `TWork` index
for each signature separately.

Let's try a few examples, to see what happens.

### Example 1 Decision Tree Creation

In this example, we'll take a very simple set of signatures, and work out a full decision tree
that definitely resolves or rejects.

```
A: (int8, int8) -> int8
B: (int16, int16) -> int16
C: (float32, float32) -> float32
```

#### Node 0

```
S_working = {A, B, C}
Symbolic_args = [[int8, int8], [int16, int16], [float32, float32]]
Equations = [S.TInp[0].matches(TInp[0]),
             S.TInp[1].matches(TInp[1]]

Candidate 1: hash-map of TInp[0].typeid
S_working hash-map: {int8: {A}, int16: {B}, float32: {C}, <default>: {}}
Candidate 2: hash-map of TInp[1].typeid
S_working hash-map: {int8: {A}, int16: {B}, float32: {C}, <default>: {}}
```

The scores of both are equal (they produce the exact same discrimination between signatures), so
we go with candidate 1. No new equations are generated, so that leaves us with just one more equation
to process.

#### Node 1 <int8< Node 0

All but one signature has been culled. There is only one more possible candidate to check, the equation
for the second argument's type.

```
S_working = {A}
Symbolic_args = [[int8, int8], culled, culled]
Equations = [S.TInp[1].matches(TInp[1]]

Candidate 1: hash-map of TInp[1].typeid
S_working hash-map: {int8: {A}, <default>: {}}
```

#### Node 2 <int8< Node 1

```
S_working = {A}
Symbolic_args = [[int8, int8], culled, culled]
Equations = []
```

No equations are left, so we return `A` as a successful match.

#### Node 3 <int16< Node 0

```
S_working = {B}
Symbolic_args = [culled, [int16, int16], culled]
Equations = [S.TInp[1].matches(TInp[1]]

Candidate 1: hash-map of TInp[1].typeid
S_working hash-map: {int16: {B}, <default>: {}}
```

#### Node 4 <int16< Node 3

```
S_working = {B}
Symbolic_args = [culled, [int16, int16], culled]
Equations = []
```

We return `B` as a successful match.

#### Node 5 <float32< Node 0

```
S_working = {C}
Symbolic_args = [culled, culled, [float32, float32]]
Equations = [S.TInp[1].matches(TInp[1]]

Candidate 1: hash-map of TInp[1].typeid
S_working hash-map: {int16: {B}, <default>: {}}
```

#### Node 6 <float32< Node 5

```
S_working = {C}
Symbolic_args = [culled, culled, [float32, float32]]
Equations = []
```

We return `C` as a successful match.

#### Node 7 << From any empty set

We return "no match".

### Example 1 Resulting Decision Tree Code

Once the decision tree is completed, it needs to be turned into something that
can be executed, likely a simple byte code. Here's the sequence of instructions
that the decision tree produces.

```python
Node0:
    goto {int8: Node1, int16: Node3, float32: Node5}
         .get(TInp[0].typeid, Node7)
Node1:
    goto {int8: Node2}
         .get(TInp[1].typeid, Node7)
Node2:
    return MatchSuccess(0) # Signature A
Node3:
    goto {int16: Node4}
         .get(TInp[1].typeid, Node7)
Node4:
    return MatchSuccess(1) # Signature B
Node5:
    goto {float32: Node6}
         .get(TInp[1].typeid, Node7)
Node6:
    return MatchSuccess(2) # Signature C
Node7:
    return MatchFailure()
```

### Example 1 Decision Tree Execution Traces

Signature `(int8, int16)`

```
Node0:
    goto {int8: Node1, int16: Node3, float32: Node5}
         .get(TInp[0].typeid, Node7) # int8
 => goto Node1
Node1:
    goto {int8: Node2}
         .get(TInp[1].typeid, Node7) # int16
 => goto Node7
Node7:
    return MatchFailure()
```

Signature `(float32, float32)`

```
Node0:
    goto {int8: Node1, int16: Node3, float32: Node5}
         .get(TInp[0].typeid, Node7) # float32
 => goto Node5
Node5:
    goto {float32: Node6}
         .get(TInp[1].typeid, Node7) # float32
 => goto Node6
Node6:
    return MatchSuccess(2) # Signature C
```

### Example 2

We add a slight wrinkle which example 1's decision tree no longer works for.

```
A: (int8, int8) -> int8
B: (int16, int16) -> int16
C: (float32, float32) -> float32
D: (int16, float32) -> float32
```

#### Node 0

```
S_working = {A, B, C, D}
Symbolic_args = [[int8, int8], [int16, int16], [float32, float32], [int16, float32]]
Equations = [S.TInp[0].matches(TInp[0]),
             S.TInp[1].matches(TInp[1]]

Candidate 1: hash-map of TInp[0].typeid
S_working hash-map: {int8: {A}, int16: {B, D}, float32: {C}, <default>: {}}
Candidate 2: hash-map of TInp[1].typeid
S_working hash-map: {int8: {A}, int16: {B}, float32: {C, D}, <default>: {}}
```

Once again, the scores are equal, so we choose candidate 1. The `int16` branch is still ambiguous,
so we link it to Node 1, and recursively apply our algorithm. Since we already checked the type id
of `TInp[0]`, it was marked as finished, and there is only one candidate left.

#### Node 3 <int16< Node 0

```
S_working = {B, D}
Symbolic_args = [culled, [int16, int16], culled, [int16, float32]]
Equations = [S.TInp[1].matches(TInp[1]]

Candidate 1: hash-map of TInp[1].typeid
S_working hash-map: {int16: {B}, float32: {D}, <default:> {}}
```

### Example 2 Resulting Decision Tree Code

We haven't worked out the full decision tree explicitly, but let's turn what we would generate
into code as we did for example 1.

```python
Node0:
    goto {int8: Node1, int16: Node3, float32: Node6}
         .get(TInp[0].typeid, Node8)
Node1:
    goto {int8: Node2}
         .get(TInp[1].typeid, Node8)
Node2:
    return MatchSuccess(0) # Signature A
Node3:
    goto {int16: Node4, float32: Node5}
         .get(TInp[1].typeid, Node8)
Node4:
    return MatchSuccess(1) # Signature B
Node5:
    return MatchSuccess(3) # Signature D
Node6:
    goto {float32: Node7}
         .get(TInp[1].typeid, Node8)
Node7:
    return MatchSuccess(2) # Signature C
Node8:
    return MatchFailure()
```

### Example 3

Let's add two symbolic signatures so the partial ordering isn't just independent elements anymore.

```
A: (int8, int8) -> int8
B: (int16, int16) -> int16
C: (float32, float32) -> float32
D: (int16, float32) -> float32
E: (T, T) -> T
F: (S, T) -> S
```

#### Node 0

```
S_working = {A, B, C, D, E, F}
Symbolic_args = [[int8, int8], [int16, int16], [float32, float32], [int16, float32], [T, T], [S, T]]
Equations = [S.TInp[0].matches(TInp[0]),
             S.TInp[1].matches(TInp[1]]

Candidate 1: hash-map of TInp[0].typeid
S_working hash-map: {int8: {A, E, F}, int16: {B, D, E, F}, float32: {C, E, F}, <default>: {E, F}}
Candidate 2: hash-map of TInp[1].typeid
S_working hash-map: {int8: {A, E, F}, int16: {B, E, F}, float32: {C, D, E, F}, <default>: {E, F}}
```

The scores of candidates 1 and 2 are equal, so we'll proceed with candidate 1,
and consider the `int8` branch first.

#### Node 1 <int8< Node 0

When a type variable in a signature is processed, we need to handle things specially so that the
type variable is matched consistently across all the types in that signature.
One possibility is that the edge we took fully determines the type variable, in which case
we can substitute that type variable in our symbolic representation which causes further matches
against it to be treated concretely. The other is that the edge we took expands out new parameters,
in which case we need to save the type we say into a working vector and keep track of it in a symbol
table.

By taking the `int8` edge, we fully know the type of `T` in signature `E` and the type of `S` in
signature `F`. Thus those can be substituted fully in the symbolic args, and no tracking of the
type needs to be done at decision tree evaluation time.

```
S_working = {A, E, F}
Symbolic_args = [[int8, int8], culled, culled, culled, [int8, int8], [int8, T]]
Equations = [S.TInp[1].matches(TInp[1]]

Candidate 1: hash-map of TInp[1]
S_working hash-map: {int8: {A}, <default:> {F}}
```

Note that signatures `E` and `F` are eliminated from the `int8` edge due to `matches` partial ordering constraints.

#### Node 2 <int16< Node 0

```
S_working = {B, D, E, F}
Symbolic_args = [culled, [int16, int16], culled, [int16, float32], [int16, int16], [int16, T]]
Equations = [S.TInp[1].matches(TInp[1]]

Candidate 1: hash-map of TInp[1].typeid
S_working hash-map: {int16: {B}, float32: {D}, <default>: {F}}
```

Once again, signatures `E` and `F` are eliminated from the `int16` edge due to `matches` partial ordering
constraints, and signature `F` is eliminated from the `float23` edge for the same reason.

#### Node 3 <float32< Node 0

```
S_working = {C, E, F}
Symbolic_args = [culled, culled, [float32, float32], culled, [float32, float32], [float32, T]]
Equations = [S.TInp[1].matches(TInp[1]]

Candidate 1: hash-map of TInp[1].typeid
S_working hash-map: {float32: {C}, <default>: {F}}
```

#### Node 4 <default< Node 0

In taking this edge, `TInp[0]` is appended to `TWork`, and a symbol correspondence is set up in each
signature with a type variable at that position.

```
S_working = {E, F}
Symbolic_args = [culled, culled, culled, culled,
                 [T, T], {T: TWork[0]},
                 [S, T], {S: TWork[0]}]
Equations = [S.TInp[1].matches(TInp[1]]

Candidate 1: TInp[1] == TWork[0]
S_working yes: {E}, no: {F}
```

The `matches` partial ordering eliminates signature `F` from the `yes` edge.

### Example 4

Let's add in some optional types and a single dimension ellipsis to require usage of `TWork`.

```
A: (int8, int8) -> int8
B: (int8, int16) -> int16
C: (int8, int32) -> int32
D: (?int8, int8) -> ?int8
E: (?int8, int16) -> ?int16
F: (?int8, int32) -> ?int32
G: (Dims... * ?int8, int8) -> ?int8
```

#### Node 0

```
S_working = {A, B, C, D, E, F, G}
Symbolic_args = [[int8, int8], [int8, int16], [int8, int32],
                 [?int8, int8], [?int8, int16], [?int8, int32],
                 [Dims... * ?int8, int8]]
Candidate 1: hash-map of TInp[0].typeid
S_working hash-map: {int8: {A, B, C}, option: {D, E, F}, <dim>: {G}, <default>: {}}
Candidate 2: hash-map of TInp[1].typeid
S_working hash-map: {int8: {A, D, G}, int16: {B, E}, int32: {C, F}, <default>: {}}
```

While `G` matches when `option` is seen in candidate 1, the symbolic substition of no dimensions
into its signature produces `[?int8, int8]` which exactly matches a  more specific signature. Therefore
it is removed from that set.

Candidate 1 can instantly narrow one case down to a singleton set `G`, so it probably deserves
a higher score than candidate 2.

#### Node 1 <int8< Node 0

```
S_working = {A, B, C}
Symbolic_args = [[int8, int8], [int8, int16], [int8, int32],
                 culled, culled, culled,
                 culled]
Candidate 1: hash-map of TInp[1].typeid
S_working hash-map: {int8: {A}, int16: {B}, int32: {C}, <default>: {}}
```

Only one choice here, which fully resolves the signature.

#### Node 2 <option< Node 0

As part of traversing an `option` link, the type contained in the `option` is appended to
`TWork`. During execution of the decision tree, this is done as an operation when arriving at
node 2.

```
S_working = {D, E, F}
Symbolic_args = [culled, culled, culled,
                 [?int8, int8] + TWork [int8], [?int8, int16] + TWork [int8], [?int8, int32] + TWork [int8],
                 culled]
Candidate 1: hash-map of TInp[1].typeid
S_working hash-map: {int8: {D}, int16: {E}, int32: {F}, <default>: {}}
Candidate 2: hash-map of TWork[0].typeid
S_working hash-map: {int8: {D, E, F}, <default>: {}}
```

Candidate 1 exactly discriminates the signatures, while candidate 2 provides no additional information.

### Example 5

Let's do a gradient function.

```
A: (D0 * float32) -> D0 * float32
B: (D1 * D0 * float32) -> D1 * D0 * 2 * float32
C: (Fixed**N * float32) -> Fixed**N * N * float32
```

#### Node 0

```
S_working = {A, B, C}
Symbolic_args = [[D0 * float32], [D1 * D0 * float32], [Fixed**N * float32]]
Candidate 1: hash-map of TInp[0].typeid
S_working hash-map: {fixed: {A, B, C}, <default>: {}}
```

The candidate doesn't shrink any signature subsets, but does produce more information.

#### Node 1 <fixed< Node 0

The first fixed dimension gets stripped off the argument, and its size gets appended to the
`IWork` vector while its element type gets appended to `TWork`. The representation of `IWork`
for each signature keeps track of the type variable it belongs to.

```
S_working = {A, B, C}
Symbolic_args = [[D0 * float32] + IWork [D0] + TWork [float32],
                 [D1 * D0 * float32] + IWork [D1] + TWork [D0 * float32],
                 [Fixed**N * float32] + IWork [??] + TWork [Fixed**(N-1) * float32]]
Candidate 1: hash-map of TWork[0].typeid
S_working hash-map: {float32: {A}, fixed: {B, C}, <default>: {}}
Candidate 2: hash-map of IWork[0]
S_working hash-map: {<default>: {A, B, C}}
```

Candidate 1 clearly wins.

#### Node 2 <fixed< Node 1

The first fixed dimension gets appended to `IWork` and `TWork` as in Node 1.

```
S_working = {B, C}
Symbolic_args = [culled,
                 [D1 * D0 * float32] + IWork [D1, D0] + TWork [D0 * float32, float32],
                 [Fixed**N * float32] + IWork [??, ??] + TWork [Fixed**(N-1) * float32, Fixed**(N-2) * float32]]
Candidate 1: hash-map of TWork[1].typeid
S_working hash-map: {float32: {B}, fixed: {C}, <default>: {}}
Candidate 2: hash-map of IWork[0]
S_working hash-map: {<default>: {A, B, C}}
Candidate 3: hash-map of IWork[1]
S_working hash-map: {<default>: {A, B, C}}
```

Candidate 1 fully distinguishes the signatures, so we're done.

### Example 5

A simple broadcasting ufunc.

```
A: (Dims... * int8, Dims... * int16) -> Dims... * int8
B: (Dims... * int16, Dims... * int32) -> Dims... * int16
```

#### Node 0

```
S_working = {A, B}
Symbolic_args = [[Dims... * int8, Dims... * int16],
                 [Dims... * int16, Dims... * int32]]
Candidate 1: hash-map of TInp[0].typeid
S_working hash-map: {any dim: {A, B}, int8: {A}, int16: {B}, <default>: {}}
Candidate 2: hash-map of TInp[1].typeid
S_working hash-map: {any dim: {A, B}, int16: {A}, int32: {B}, <default>: {}}
```

Both candidates produce the same level of discrimination, so we take the first one.

#### Node 1 <any dim< Node 0

Since all the symbolic args are `Dims...`, an ellipsis, all dimensions get consumed, and
the resulting dim fragment is saved in `TWork` along with the scalar type after the dimensions.

```
S_working = {A, B}
Symbolic_args = [[Dims... * int8, Dims... * int16] + TWork [Dims..., int8],
                 [Dims... * int16, Dims... * int32] + TWork [Dims..., int16]]
Candidate 1: hash-map of TInp[1].typeid
S_working hash-map: {fixed: {A, B}, int16: {A}, int32: {B}, <default>: {}}
Candidate 2: hash-map of TWork[1].typeid
S_working hash-map: {int8: {A}, int16: {B}, <default>: {}}
```

If we had to keep going to distinguish between the signatures, we would consume the `Dims...` for the
second argument, and then confirm that it broadcasts against what was already found for `Dims...` in
`TWork[0]`. To track this information, the symbolic information for each signature needs to also be tracking
a type variable symbol table, which maps each symbol name to its location in the various working vectors.
This allows for confirming that ellipsis type variables match dims, and that regular type variables match
across the arguments.

### Example 6

Match some tuple types, mixed up with some dimensions

```
A: (3 * (int8), (int8, int16)) -> void
B: (3 * (int8, int8), (int16, int16)) -> void
C: (3 * (int8, int8, int8), (int16, int32)) -> void
D: (N * (int8, int8), (int16, int16)) -> void
```

#### Node 0

```
S_working = {A, B, C, D}
Symbolic_args = [[3 * (int8), (int8, int16)],
                 [3 * (int8, int8), (int16, int16)],
                 [3 * (int8, int8, int8), (int16, int32)],
                 [N * (int8, int8), (int16, int16)]]
Candidate 1: hash-map of TInp[0].typeid
S_working hash-map: {fixed: {A, B, C, D}, <default>: {}}
Candidate 2: hash-map of TInp[1].typeid
S_working hash-map: {tuple: {A, B, C, D}, <default>: {}}
```

Neither candidate provides any distinguishing power. In cases like this, it feels tempting to
evaluate both decision trees, then pick the best one, instead of using a heuristic score for
a greedy construction. Let's just proceed with candidate 1.

#### Node 1 <fixed< Node 0

The fixed dimension gets pulled apart into `IWork` and `TWork`.

```
S_working = {A, B, C, D}
Symbolic_args = [[3 * (int8), (int8, int16)] + IWork [3] + TWork [(int8)],
                 [3 * (int8, int8), (int16, int16)] + IWork [3] + TWork [(int8, int8)],
                 [3 * (int8, int8, int8), (int16, int32)] + IWork [3] + TWork [(int8, int8, int8)],
                 [N * (int8, int8), (int16, int16)] + IWork [N] + TWork [(int8, int8)]]
Candidate 1: hash-map of IWork[0]
S_working hash-map: {3: {A, B, C}, <default>: {D}}
Candidate 2: hash-map of TWork[0].typeid
S_working hash-map: {tuple: {A, B, C, D}, <default>: {}}
Candidate 3: hash-map of TInp[1].typeid
S_working hash-map: {tuple: {A, B, C, D}, <default>: {}}
```

Candidate 1 is doing the most discrimination here. Notice that `D` got culled from the 3 case due
to `matches` partial ordering.

#### Node 2 <3< Node 1

```
S_working = {A, B, C}
Symbolic_args = [[3 * (int8), (int8, int16)] + IWork [3] + TWork [(int8)],
                 [3 * (int8, int8), (int16, int16)] + IWork [3] + TWork [(int8, int8)],
                 [3 * (int8, int8, int8), (int16, int32)] + IWork [3] + TWork [(int8, int8, int8)],
                 culled]
Candidate 1: hash-map of TWork[0].typeid
S_working hash-map: {tuple: {A, B, C}, <default>: {}}
Candidate 2: hash-map of TInp[1].typeid
S_working hash-map: {tuple: {A, B, C}, <default>: {}}
```

No additional candidates were generated by the last choice, so we're once again left with no
distiguishing features at the level where we make the choice. Continue with candidate 1.

#### Node 1 <tuple< Node 2

The tuple gets pulled apart into `IWork` (tuple size) and `TWork` (all the types).

```
S_working = {A, B, C}
Symbolic_args = [[3 * (int8), (int8, int16)] + IWork [3, 1] + TWork [(int8), int8],
                 [3 * (int8, int8), (int16, int16)] + IWork [3, 2] + TWork [(int8, int8), int8, int8],
                 [3 * (int8, int8, int8), (int16, int32)] + IWork [3, 3] +
                                                            TWork [(int8, int8, int8), int8, int8, int8],
                 culled]
Candidate 1: hash-map of IWork[1]
S_working hash-map: {1: {A}, 2: {B}, 3: {C}, <default>: {}}
Candidate 2: hash-map of TInp[1].typeid
S_working hash-map: {tuple: {A, B, C}, <default>: {}}
```

The freshly expanded types in `TWork` can't be relied upon until their count is locked down, which means
either we require the next branch be based on that size, or we come up with an alternate representation
of the working types. For the latter, could introduce a `TVecsWork` which has vectors of types, instead
of types directly. In fact, the latter could be raw pointers directly into the tuple/struct types that
were expanded, making their expansion cheaper.

At this point, we've fully discriminated between the signatures, so we're done. We wouldn't be done
if the second parameter wasn't identical between `B` and `D`, because the size of the outermost dimension
of the first parameter would no longer discriminate between them.
