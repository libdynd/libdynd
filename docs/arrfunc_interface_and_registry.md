# DyND Array Functions (ArrFuncs) Interface

DyND arrfuncs are a recipe for computation in DyND. They are constructed with a make function, unique for each kind of arrfunc, which
may have arguments itself. A DyND arrfunc is called with array arguments and a special array, possibly null, called "kwds" that
represents dynamic keyword arguments.

Arrfunc execution happens in two stages. The first stage, instantiation, gets the type/arrmeta of the array arguments and the values of the dynamic keyword arguments, and has the opportunity to bake that information down into a "ckernel" that will repeatedly run on multiple sets of array arguments. The second stage executes that ckernel on a set of pointers to array data.

## Make Arguments versus Keyword Arguments

Most arguments to an arrfunc could be either arguments to make or passed to operator() as keywords. Our design choice is that there should
be as few arguments to make as possible, and that keywords are preferable. A rule of thumb is that a keyword argument should be used if the type of
the argument is language-independent. For instance, an arrfunc, an array, or some simple value should all be keyword arguments. A different
case is functor_arrfunc, which takes in a C++ function pointer or callable object. This should be an argument in the make signature.
In Python, functor_arrfunc would have a make signature that takes in a Python function or callable.

## Arrfuncs in C++

A common idiom for writing an arrfunc in C++ is the following. Define a C++ function that will be made into a functor_arrfunc object.
Use that functor_arrfunc object to get a higher-level arrfunc, like elwise or neighborhood or reduce, that may also have dynamic
parameters.

We need a way to create simple instances of arrfuncs like the above given a C++ function. For instance, I may have a function 

```
int calc(int x, int y) {
  return x + 2 * y;
}
```

that I'd like an elwise version of. It seems we should be able to do something like

nd::arrfunc elwise_calc = make_elwise_arrfunc(calc);

A bit less trivial is something like neighborhood that has keyword arguments like shape and offset. In that case, if I have some function strided_calc, I should be able to have an arrfunc with a simple interface with those arguments unpacked. I think this is easy enough to do if we restrict each argument to a fixed number of source argument (e.g., multiple dispatch is for different types, not variable number of sources). As the function prototypes will know the keyword names, we can add methods like

```
template <typename K0, ..., typename KN>
nd::array operator ()(const nd::array &a0, ..., const nd::array &an, const K0 &k0, ..., const KN &kn) {
    if (n != number_of_fixed_src) {
      raise error
    } else {
      package k0, ..., kn into kwds, then call the arrfunc
    }
  }
};
```

(Actually, we probably don't even have to restrict arrfuncs to a fixed number of srcs, we just need to query different prototypes.)

## Default Keyword Arguments

Keyword arguments need to have default values. Also, we need to be able to set it up so that keyword arguments can take values computed using other arrfuncs. An example is a keyword argument like "val_max", being the maximum value in an array. The arrfunc should be able to call max to work this out by default.

## Low-Level C++ Access

Most functions that we make arrfuncs from in DyND, like sin(x), cos(x), ..., should be available as pure C++ functions operating on static types, as well as arrfuncs. This would allow users of the library to simply call those functions easily within their own C++ functions that may be passed to functor_arrfuncs. Right now, many of these functions are in source (*.cpp) files, which makes that impossible.

Because we want to support CUDA, this access probably needs to be in a proper function definition, not through an instance of arrfunc.

## Arrfuncs Composed Entirely of Other Arrfuncs

How does one define an arrfunc that is very simple, being composed just of other arrfuncs. Maybe I want an arrfunc like (x * y - z), where x, y, z are all arrays. I should be able to very easily define that as an arrfunc.

## Callable versus Arrfunc

DyND currently has both callables and arrfuncs. What is the purpose of one versus the other?

## Pluggable Types

If someone implements a new type, like float4096, how can an arrfunc like sin(x) be extended to take advantage of that?

## Arrfuncs in Python

It should be very easy to take a C++ arrfunc and bring it to the Python level. Perhaps we can have two interfaces to this, one through the standard Python C API and the other through Cython.
