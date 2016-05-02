DyND Developer Guide
====================

This document is an introduction to DyND for any developers
who would like to understand the inner workings of the code,
and possibly contribute changes to the DyND codebase.
While aspects of its design have solidified, many parts of
the code and design are still undergoing large changes, driven
by adding new functionality and connecting DyND to
other systems.

As you explore the system, be aware that the code has gone
through several design iterations, and there is code that
used to work but doesn't at the moment because not everything
in the system has caught up yet. These parts of the code
should gradually be updated or discarded.

The Build System
----------------

To dig into the code, you probably want to start by getting
it building and the tests running on your system. While you can
work with just the pure C++ DyND library, we recommend you get
the Python bindings and build and run both simultaneously with
`python setup.py develop`. On Mac/Linux, this creates a single
directory which builds both together and loads in the target
Python with no additional steps, and on Windows, this creates
a Visual Studio solution that has the same effect.

The instructions to get the source code and build it is available in the
[build/install guide for dynd-python](https://github.com/libdynd/dynd-python/blob/master/BUILD_INSTALL.md).

Running The Tests
-----------------

During development of DyND, running the test suite should be
a reflex action. Except in rare cases, both the C++ and
the Python test suites should be passing 100% on OS X, Windows,
and Linux.

The C++ tests are using the google test framework, and to
run them you simply run the program 'test_libdynd' which is built
from the libdynd/tests subdirectory. On Windows MSVC, right
click on the test_libdynd project and choose "Set as StartUp
Project", so when you hit play, the tests will run.

When you're building the Python bindings as well as is recommended,
the Python tests can be run with a command like

    python -c "import dynd;dynd.test()"

Background Introductory Material
--------------------------------

To get some insight into some of the design choices made in DyND,
there are some introductory slides created in
[Jupyter notebook](http://jupyter.org/) form, and
presentable using the [RISE](https://github.com/damianavila/RISE)
Jupyter extension.

The first set of such slides is about
[how DyND views memory](http://nbviewer.ipython.org/github/libdynd/libdynd/blob/master/docs/intro-01/HowDyNDViewsMemory.ipynb).

Three video talks about DyND are available. One was presented at the
[AMS in January 2013](https://ams.confex.com/ams/93Annual/webprogram/Paper224314.html),
another was presented at
[SciPy in June 2013](https://conference.scipy.org/scipy2013/presentation_detail.php?id=175), and the third was presented as a lightning talk at
[EuroSciPy in August 2014](https://www.euroscipy.org/2014/schedule/euroscipy-2014-general-sessions/).
DyND was also mentioned in a Blaze talk at
[SciPy in July 2014](https://conference.scipy.org/scipy2014/schedule/presentation/1717/).

* [AMS January 2013 DyND Talk](https://ams.confex.com/ams/93Annual/flvgateway.cgi/id/23375?recordingid=23375)
* [SciPy June 2013 DyND Talk](https://www.youtube.com/watch?v=BduIKN5mgvU)
* [SciPy July 2014 Blaze Talk](https://www.youtube.com/watch?v=9HPR-1PdZUk)
* [EuroSciPy August 2014 DyND Lightning Talk](https://www.youtube.com/watch?v=VZ7enVMNB84#t=198)
* [SciPy July 2015 DyND Talk](https://www.youtube.com/watch?v=ZSWcX6yaQrQ)

Source Code Directory Layout
----------------------------

The main directories to look at within DyND are the `src`
and `include` directories. These contain the main source
code and header files.

The source files are mostly distributed in subdirectories
of `src/dynd`, with a few that are the main objects or
not easily categorized in `src/dynd` itself. The same
applies to the headers in the `include/dynd` directory. In
general, if a header .hpp file has a corresponding .cpp file,
it goes into the `src/*` directory corresponding
to `include/*`.

### Main Source

In the
[``include/dynd``](https://github.com/libdynd/libdynd/tree/master/include/dynd)
directory, the files implementing the main DyND objects are
[``type.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/type.hpp)
and
[``array.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/array.hpp).
The file
[``config.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/config.hpp)
has some per-compiler logic to enable/disable various supported
C++ features, define some common macros, and declare the
version number strings.

### Types Source

All the DyND types are presently located in
[``include/dynd/types``](https://github.com/libdynd/libdynd/tree/master/include/dynd/types).
The file
[``include/dynd/type.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/type.hpp)
basically contain a smart pointer which wraps instances of type
[``base_type *``](https://github.com/libdynd/libdynd/blob/master/include/dynd/types/base_type.hpp)
in a more convenient interface. The file
[``include/dtypes/type_id.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/types/type_id.hpp)
contains a number of primitive enumeration types and some
type_id metaprograms.

There are a number of ``base_<kind>_dtype`` dtype classes
which correspond to a number of the dtype 'kinds'. For
example, if a dtype is of the ``dim`` kind, then it must inherit from
[``base_dim_type``](https://github.com/libdynd/libdynd/blob/master/include/dynd/types/base_dim_type.hpp).

The code which consumes and produces Blaze datashapes is in
[``include/dynd/types/datashape_parser.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/types/datashape_parser.hpp)
and
[``include/dynd/types/datashape_formatter.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/types/datashape_formatter.hpp).

See [the DataShape documentation](http://datashape.pydata.org/)
for more general information about datashape.

### Arrays Source

The DyND
[``nd::array``](https://github.com/libdynd/libdynd/blob/master/include/dynd/array.hpp)
object is a smart-pointer object which wraps the lower-level types, memory
blocks, and arrmeta handling into an object with a more convenient
interface. This object has reference semantics, copying one ``nd::array``
to another creates another reference to the same object. It also has
many constructors from C++ objects, which create new ``nd::array``
objects of the appropriate type, with a copy of the value provided.

Some example code creating and modifying an ``nd::array``:

```c++
// Initialize an array using C++11 initializer list
nd::array a = {1.5, 2.0, 3.1};
cout << a << endl;
// Assign one value
a(1).vals() = 100;
cout << a << endl;
// Assign a range (slightly more efficient with vals_at)
a.vals_at(irange() < 2) = {9, 10};
cout << a << endl;
```

Output:

```
array([1.5,   2, 3.1],
      type="3 * float64")
array([1.5, 100, 3.1],
      type="3 * float64")
array([  9,  10, 3.1],
      type="3 * float64")
```

### Func Source

The
[``func``](https://github.com/libdynd/libdynd/tree/master/include/dynd/func)
namespace is where the
[``nd::arrfunc``](https://github.com/libdynd/libdynd/blob/master/include/dynd/func/arrfunc.hpp)
and
[``nd::callable``](https://github.com/libdynd/libdynd/blob/master/include/dynd/func/callable.hpp)
objects are defined, which provide
array function abstractions. The ``nd::arrfunc`` is being actively
developed, and the plan is for ``callable`` to go away once all its
functionality can be superceded.

From C++, the easiest way to create an arrfunc is with
[``make_apply_arrfunc``](https://github.com/libdynd/libdynd/blob/master/include/dynd/func/apply_arrfunc.hpp),
which uses template metaprogramming to automatically generate the
required dynamic type information, ckernels, and glue functions that
an ``nd::arrfunc`` requires.

Another useful operation, after you've created a scalar arrfunc, is
to lift the arrfunc into something like a NumPy ufunc with
[``lift_arrfunc``](https://github.com/libdynd/libdynd/blob/master/include/dynd/func/lift_arrfunc.hpp).
Another example that has a more interesting function signature is
[``make_rolling_arrfunc``](https://github.com/libdynd/libdynd/blob/master/include/dynd/func/rolling_arrfunc.hpp),
which applies a provided arrfunc to every interval of an array in
a rolling fashion.

### CKernels Source

Before an ``nd::arrfunc`` is executed, it first gets lowered into
a ckernel. This low level abstraction defines an array-oriented,
possibly hierarchical way to combine code and data.

CKernels in DyND are mostly implemented in the
[``include/kernels``](https://github.com/libdynd/libdynd/tree/master/include/dynd/kernels)
directory. All kernels get built via the
[``ckernel_builder``](https://github.com/libdynd/libdynd/blob/master/include/dynd/kernels/ckernel_builder.hpp)
class. These kernels are intended to be cheap to construct and
execute when they are small, but scale to hold acceleration
structures and other information when what they are computing
is more complex.

An example trivial case is a kernel which adds two floating
point numbers, which as a ckernel will consist of just the
``addition`` function pointer and a NULL destructor. In
simple cases like this, and with a small bit of additional
composition such as operating on a dimension, creating and
executing the ckernel does not use any dynamic memory allocation.

A more complicated case is a kernel which evaluates an
expression on values in a ``struct``. Such a ckernel may contain
a bytecode representation of the expression, information
about field offsets within the struct, and child ckernels
for the component functions of the expression.

There are presently three kinds of kernels that are defined
and used within DyND: ``expr_single_t``, ``expr_strided_t``,
and ``expr_predicate_t``, defined in
[ckernel_prefix.hpp](https://github.com/libdynd/libdynd/blob/master/include/dynd/kernels/ckernel_prefix.hpp).
In all three cases there is an
array of ``src`` inputs, and a single ``dst`` output, which
is a boolean value returned in an ``int`` for the predicate
case.

The ``expr_single_t`` ckernel function prototype is for
the case where the kernel needs to be called on different
memory locations, one at a time. Implementations of it
can focus on doing their job just once with as little overhead
as possible.

The ``expr_strided_t`` ckernel function prototype is for
evaluating the function on many data values with a constant
stride between them. This is a common case in DyND, and
in general in systems based on arrays. Implementations may
check the stride for the contiguous case, to do SIMD
execution, for example. There is a constant
``DYND_BUFFER_CHUNK_SIZE`` defined, which is the size
DyND typically uses to chunk, so having a fast path for
this size, or blocking in chunks of this size may be
advantageous as well.

The ``expr_predicate_t`` ckernel function prototype is for
things like comparisons, to allow the boolean result to
be returned directly in a register instead of through a
memory address. It's used by functions such as ``sort``,
where many comparisons on different memory addresses are
performed.

Unary kernels, where the number of ``src`` inputs is one,
form the backbone of being able to copy values between
types. Some basic utilities for defining and working
with them are in
[``assignment_kernels.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/kernels/assignment_kernels.hpp).

General-purpose helper classes for defining ckernels are in
[``ckernel_builder.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/kernels/ckernel_builder.hpp),
implemented to work for both regular CPU and CUDA via some
preprocessor code.

### Memory Blocks Source

Dynamic ``nd::array`` data in DyND is stored in memory blocks,
which are in the
[``dynd/memblock``](https://github.com/libdynd/libdynd/tree/master/include/dynd/memblock)
directory. The base class is defined in
[``memory_block.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/memblock/memory_block.hpp),
and the different memory blocks serve the needs of
various dtypes.

The
[``array_memory_block``](https://github.com/libdynd/libdynd/blob/master/include/dynd/memblock/array_memory_block.hpp)
is for nd::array instances. The file
[``dynd/array.hpp``](https://github.com/libdynd/libdynd/blob/master/include/dynd/array.hpp)
defines an object which contains an ``array_memory_block`` and
defines operations on it. In typical
usage, this contains the type, arrmeta, and a reference+pointer
to memory holding the data. As an optimization, it is possible
to allocate an ``array_memory_block`` with extra memory at
the end to store the data with fewer memory allocations, in
which case the reference to the data contains NULL.

The
[``fixed_size_pod_memory_block``](https://github.com/libdynd/libdynd/blob/master/include/dynd/memblock/fixed_size_pod_memory_block.hpp)
is very simple, for POD (plain old data) types that have a fixed
size and alignment.

The
[``pod_memory_block``](https://github.com/libdynd/libdynd/blob/master/include/dynd/memblock/pod_memory_block.hpp)
provides a simple pooled memory
allocator that doles out chunks of memory sequentially
for variable-sized data types. Due to the way nd::array
uses views, similar to NumPy, once some memory has been
allocated within the pod memory block, it cannot be
deallocated until the whole memory block is freed.
A small exception is for reusable temporary buffers, and a
``reset`` operation is provided to support that use case.

The
[``zeroinit_memory_block``](https://github.com/libdynd/libdynd/blob/master/include/dynd/memblock/zeroinit_memory_block.hpp)
is just like pod_memory_block, but initializes the memory it
allocates to zero before returning it. Types have a flag
indicating whether they require zero-initialization or not.

The
[``external_memory_block``](https://github.com/libdynd/libdynd/blob/master/include/dynd/memblock/external_memory_block.hpp)
is for holding on to data owned
by a system external to DyND. For example, in some cases
DyND can directly map onto the string data of Python's
immutable strings, by using this type of memory block
and flagging the nd::array as immutable.
