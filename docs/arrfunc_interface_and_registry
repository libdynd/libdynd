# DyND Array Functions (ArrFuncs)

DyND arrfuncs are a recipe for computation in DyND. They are constructed with a make function, unique for each kind of arrfunc, which
may have arguments itself. A DyND arrfunc is called with array arguments and a special array, possibly null, called "kwds" that
represents dynamic keyword arguments.

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

int calc(int x, int y) {
  return x + 2 * y;
}

that I'd like an elwise version of. We should be able to do something like

nd::arrfunc elwise_calc = elwise(calc);

