# Proposed Outline of Fixes For Operators on DyND's Arithmetic Types

This is a tentative plan for how we can go from the proliferation bugs we currently have with our arithmetic operations to having all arithmetic operations work smoothly.
There are several pieces of this plan that may or may not be implimented at all.
This document is meant to serve primarily as a source of direction for future work.

As things stand, though the goal is to allow all reasonable operations for DyND arrays, this hinges on getting all the corresponding operations to work smoothly for the corresponding arithmetic types.
The mechanism for defining arithmetic operators has already been mostly reduced to a matter of applying a few macros in the right places, so, in most respects, if everything is working well with scalars, it will be easy to get everything working with arrays.
That said, many of the existing issues with arithmetic operators on arrays are greatly exacerbated by the fact that the arithmetic kernels lie in the `dynd::nd` namespace.
The current division of namespaces makes it so that any operator that does not have a version that is usable without implicit casts becomes ambiguous.
It also makes it so that logical operators, that otherwise might have triggered an explicit cast to bool, are ambiguous as well.
Regrouping the namespaces to separate the kernels from the nd namespace, though not strictly necessary for this proposal to work, would make things a lot easier.
One possible regrouping would be to move the existing functionality for creating and manipulating callables into a namespace within the `dynd` namespace.

In moving toward a better state for DyND's scalar types, one primary architectural concern is extendability.
Since one of the primary purposes of DyND is to make it easier for users to define and use their own types within N-dimensional array objects, any system for defining types beyond the built-in numeric types in C++ should be both robust and easy to extend.
It should be designed so that users can, with a minimal ammount of effort, adapt an existing class representing some sort of integer, floating point, or complex number so that it can be used seamlessly, not only within DyND arrays, but in conjunction with DyND's scalar types.
In the relatively near future it may make sense for DyND to provide (by writing our own versions or wrapping another library) arbitrary precision signed and unsigned integer types, and arbitrary precision real and complex floating point types.
The system we ultimately adopt should treat arbitrary-precision cases sensibly.

Another issue this proposal seeks to address is the organization of the templates that define operations for given sets of types.
Currently, it's not clear where such templates should go other than `config.hpp`.


## Options For Implicit Casts

With regards to the fixes needed for the scalar types, one of the primary issues that must be solved is which types are implicitly convertible to each other.
There are two possible approaches to this.
We can follow the existing approach for C++ scalars and make everything implicitly convertible to everything with the single exception that complex numbers are not allowed to be implicitly cast to other "real valued" scalar types.
An alternative route is to be more conservative about implicit conversions and define minimal wrappers for builtin scalar types that limit the implicit conversions that would otherwise be allowed with C++ arithmetic types.
Many of the elements of this document are equally applicable in both cases, so a more thorough discussion regarding the benefits and costs of providing wrappers for the builtin scalar types is reserved for later in this document.

Liberally allowing implicit casts between all non-complex arithmetic types has some important consequences.
One is that conversions between all currently existing types must be defined and marked as implicit.
Another is that any additional types, whether they be user-defined or included in DyND, to maintain consistent behavior between all existing scalar types, should also allow implicit casts to and from one another and all currently existing types.
In order to reduce the amount of redundant and meaningless code involved in this process, methods for defining all these conversions must also be considered.
In order to avoid further obfuscation of the errors resulting from currently missing operators, these implicit conversions cannot be enabled until all needed operators for all currently existing types are working properly.


## Which Operations Should Be Defined

For the most part, the operators that should be allowed with our scalar types match the ones already provided for the existing C++ arithmetic typs.
The primary exception are types defined by the `std::complex` template which cannot be converted to bool and do not allow for mixed-type arithmetic with anything other than the type used as their template parameter.
Though C++ does not allow it, our arithmetic types should allow a broader range of arithmetic operations involving mixes of complex and real types.
In particular, all arithmetic operators (unary and binary `+`, unary and binary `-`, binary `*`, and `/`), all logical operators (`!`, `&&`, and `||`), and all equality comparisons (`==` and `!=`) should apply to any combination of arithmetic types.
All comparison operators (`<`, `<=`, `>`, and `>=`) should apply to all integral and floating point types, but not to complex numbers.
The mod operator (`%`) and all the bitwise operators (`~`, `&`, `|`, `^`, `<<`, and `>>`) should apply exclusively to integral types.

The corresponding in-place versions of these operators should all apply to the same types as their non-in-place variants except that a given in-place operator should not be usable with a floating point or integer value on the left and a complex value on the right.

In conjunction with the existing logical and bitwise operators, it would be good to include several other callables that represent operations that users may want to perform that may be better done at the kernel level as opposed to being done by composing logical operations on arrays.
Some possible candidates include a logical xor, logical and bitwise negations of the and and or operators, and checks to see if a given entry is zero or is nonzero.
Each of these additional operators should also have a corresponding in-place version.
Each of these operators, along with their in-place versions should be enabled according to how the operator is classified, i.e. as either an equality test, logical operator, bitwise operator, logical in-place operator, or bitwise in-place operator.

In order to address the lack of an ordering for complex numbers and to provide an ordering function that does not suffer from many of the limitations present with the current comparison operations for floating point numbers, a three state total ordering callable should also be included.
This total ordering function should follow the comparison system outlined for total ordering in the C++ standard proposal at http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4367.html.

It would also be nice to have a lexicographic sorting function that sorts values along a given set of axes lexicographically.

Total orderings and other custom functions should be implementable by user-defined types via template specializations.
The definitions of the needed specializations of these functions should be implimented alongside the classes to which they apply.
A new header should be created to provide forward declarations for these new functions.

In keeping with the precedent set by Python, C++, Julia, and others, we have opted to disallow comparisons for complex types.
This is primarily to avoid all the potential ambiguities that arise when comparing and sorting complex numbers.
Users who wish to compare complex numbers should use the total ordering callable instead.


## Organization

## Base Airthmetic Class

Since logical operators should apply to all arithmetic types, they should be defined once in a template superclass that can be used to define other classes (e.g. `class my_type : public arithmetic_type<my_type, other_args...>;`).
These logical operators should rely on user-defined types to provide a conversion to bool and nothing else.
Explicitly defining all the logical operators here should not be necessary if the operator kernels are separated out of the `dynd::nd` namespace.

This base arithmetic class should also provide a unary plus operator that does nothing other than return the operand by value.
Types that subclass from the base arithmetic class should also be automatically registered as DyND arithmetic types in the `dynd::is_arithmetic` type trait.
The existing templates for lcast and rcast arithmetic should be moved out of `config.hpp` and into the header for this class.
They should be modified so that they only apply to the cases where an arithmetic operator is applied to a floating point number and an integer (in any order).
Any arithmetic operation involving a floating point number and an integer should return a floating point number.
The only exceptions there should be the in-place and logical operators.

### Integer Class

There should be a base integer class template that subclasses from the base arithmetic class and provides a similar subclass-able API to specific integer types.
Templated assignments and implicit conversions from float to integer and from another integer type to a given integer that use a generic (though slow) algorithm to convert a floating point number to an integer.
Such an algorithm may require that specific operations or static constants be defined in each integer class that subclasses from the generic integer type.
These requirements should be documented somewhere.
The generic integer template should automatically register itself as an integer with `dynd::is_integral`.
Generic comparison operators can also be provided (though having generic algorithms for comparisons here is currently of lesser importance).

### Floating Point Class

There should be a floating point class template analoguous to the integer class that provides assignments and implicit conversions between floating point types and the conversiosn from an integer to a floating point type.
It should also register any subclass of itself as a floating point number with `dynd::is_floating_point`.

### Complex Class

The complex number class template should subclass from the base arithmetic class and should template on any floating point value.
It should provide arithmetic operations that allow integers or floating point numbers to be used with complex types.
It should provide assignments and implicit conversions from integers and floating point numbers that rely on the assignment and implicit conversions from a given type to the floating point type used as the template parameter for the complex type.
Care should be taken to ensure that operators that are not defined for `std::complex` do not become defined when users include the `dynd` namespace; this can be done via careful use of templates.

### Specific types

Each type should provide an explicit conversion to bool so that the logical operators can be defined properly.
For operators to work with mixed types, each individual type should only be required to implement the needed operators for itself and implement whatever else is needed for the generic conversions defined by its superclass.
Particular operations that are defined by the generic superclasses should be override-able by specific better-optimized operators, conversions, assignments, etc. defined within a particular type.
The generic conversion algorithms used in the generic integer and floating point classes should have specific and minimal requirements that they impose on subclasses.


## Fixing Implicit Conversions For Builtin Types

A somewhat more radical route for getting everything working with scalars would be to make it so that wrapper classes for builtin types are provided that limit the implicit conversions allowed between arithmetic types.
For the builtin types, this could be done by 
This flexibility would make several things a lot easier in terms of maintainability and extendability, but it may not be a desirable interface.

For extendability, a template could be made to test whether a given class is implicitly constructible from another.
This could then be used to forward implicit conversions from one custom arithmetic type to another.
The generic conversion algorithms could be applied implicitly only whenever an implit conversion from an input type to a type in a given parameter pack of types exists.
This would allow implicit conversions to be propegated from one type to another within our scalar type system in a sensible and controlled manner without requiring that each type be aware of all the others.
Each type would then be able to specify other types that it is implicitly constructible from and inherit the implicit conversions defined for those types.
One reasonable way to decide which conversions should be implicit would be to say that any implicit conversion should not raise an exception.

It may be possible to provide an implicit conversion propegation system for user-defined types like this without wrapping builtin scalars, but then it becomes rather ambiguous what should and shouldn't be implcitly convertible.

Wrapping the builtin types would have the positive effect that templates could be used to provide additional arithmetic operations (e.g. mixed arithmetic between `std::complex` and DyND's integer and floating point types).
It would also make it so that some of the stranger effects of C++'s arithmetic operators (e.g. how unary plus converts smaller integer types to 32 bit integers).
If we add the requirement that no two types be implicitly castable to one another, it should be possible to eliminate many of the generic operator templates that perform explicit casts on their operands to make sure the proper operator is called.
Having wrappers for builtin types would also make it so that we could define a uniform set of user-defined literals that could be used to initialize our arithmetic types.

Wrappers (or types that mimic builtin or standard types like `dynd::complex` currently does) should be implicitly convertible to and from the types they mimic.


## Other concerns

We should have some tests that use expression SFINAE to assert that, regardless of whether or not a user includes the `dynd` namespace, the needed operators for DyND's complex numbers are available and that the operators available for `std::complex` remain unchanged.

It would be very helpful to make it so that expression SFINAE can be used to determine whether or not an operator is defined.
This would make it easier for users to update the existing callables for our operators with their own types that may not need or want to have every arithmetic operation defined.
This may be difficult to do, however, without moving the kernels outside of the `dynd::nd` namespace.

Other mathematic functions, like exponentiation, should also fit into this framework.

In the generic conversion algorithms from float to int and vice versa, values of `NaN` and `Inf` should be handled sensibly.
Integers too big to be represented by a given floating point type should be translated to a positive or negative `Inf` and values of `NaN` or `Inf` should trigger an exception when converted to an integer.


## Unsolved Issues

When should we allow integer overflow in conversions and when should we throw an exception?

Should we make generic signed and unsigned integer subclasses?
Their primary purpose, for now, would be to register types that subclass from them with the corresponding type traits.

Should we make types for `complex<float16>` and `complex<float128>`?
Doing this could be a good way to test if all the operations are working properly.

Should we even allow things like `complex<int>` Such mathematical structures exist, but are seldom used.

The placement of the operators that perform casting on their operands to make sure the proper common type is used is a little ambiguous. 
These operators could be placed together with the generic arithmetic type, split up between the floating point, integer, and complex types in slightly less generic forms, or moved to their own header somewhere else.

Should we move our type traits into their own header and out of config.hpp?

How should our scalar type system interoperate with other scalar type systems in other languages?
The current design seems good for C++, but what about Python, R, Julia, Scala, and the many other languages that would benefit from having bindings for DyND?
