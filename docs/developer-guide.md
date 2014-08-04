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
the Python bindings and build and run both simultaneously. The
instructions to get the source code and build it is available
in the document

    https://github.com/ContinuumIO/dynd-python/blob/master/BUILD_INSTALL.md

Running The Tests
-----------------

During development of DyND, running the test suite should be
a reflex action. Except in rare cases, both the C++ and
the Python test suites should be passing 100% on OS X, Windows,
and Linux.

The C++ tests are using the google test framework, and to
run them you simply run the program 'test_dynd' which is built
from the dynd/tests subdirectory. On Windows MSVC, right
click on the test_dynd project and choose "Set as StartUp
Project", so when you hit play, the tests will run.

If you're building the Python bindings as well, the Python
tests can be run with a command like

    python -c "import dynd;dynd.test()"

Source Code Directory Layout
----------------------------

The main directories to look at within DyND are the 'src'
and 'include' directories. These contain the main source
code and header files.

The source files are mostly distributed in subdirectories
of ``src/dynd``, with a few that are the main objects or
not easily categorized in ``src/dynd`` itself. The same
applies to the headers in the ``include/dynd`` directory. In
general, if a header .hpp file has a corresponding .cpp file,
it goes into the ``src/*`` directory corresponding
to ``include/*``.

### Main Source

In the ``include/dynd`` directory, the files implementing the
main DyND objects are ``type.hpp`` and ``array.hpp``.
The file ``config.hpp`` has some per-compiler logic to
enable/disable various C++11 features, define some common
macros, and declare the version number strings.

### Types Source

All the DyND types are presently located in
``include/dynd/types``. The file ``include/type.hpp``
basically contain a smart pointer which wraps instances
of type ``base_dtype *`` in a more convenient interface.
The file ``include/dtypes/type_id.hpp`` contains a number
of primitive enumeration types and some type_id metaprograms.

There are a number of ``base_<kind>_dtype`` dtype classes
which correspond to a number of the dtype 'kinds'. For
example, if a dtype is of ``dim`` kind, then it
must inherit from ``base_dim_dtype``.

The code which consumes and produces Blaze datashapes is in
``include/dtypes/datashape_parser.hpp`` and
``include/dtypes/datashape_formatter.hpp``.

### Kernels Source

Kernels in DyND are mostly implemented in the
``include/kernels`` directory. All kernels get built in
the ``ckernel_builder`` class, which is defined in
``include/kernels/ckernel_builder.hpp``. These kernels
are intended to be cheap to construct and execute when they
are small, but scale to hold acceleration structures and other
information when what they are computing is more complex.

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
and ``expr_predicate_t``. In all three cases there is an
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
with them are in ``include/kernels/assignment_kernels.hpp``.

### Memory Blocks Source

Dynamic nd::array data in DyND is stored in memory blocks,
which are in the ``include/memblock`` directory. The base
class is defined in ``include/memblock/memory_block.hpp``,
and the different memory blocks serve the needs of
various dtypes.

The ``array_memory_block`` is for nd::array instances. The file
``include/array.hpp`` defines an object which contains an
array_memory_block and defines operations on it. In typical
usage, this contains the type, arrmeta, and a reference+pointer
to memory holding the data. As an optimization, it is possible
to allocate an ``array_memory_block`` with extra memory at
the end to store the data with fewer memory allocations, in
which case the reference to the data contains NULL.

The ``fixed_size_pod_memory_block`` is very simple, for data
types that have a fixed size and alignment.

The ``pod_memory_block`` provides a simple memory
allocator that doles out chunks of memory sequentially
for variable-sized data types. Due to the way nd::array
uses views, similar to NumPy, once some memory has been
allocated within the pod memory block, it cannot be
deallocated until the whole memory block is freed.
A small exception is for reusable temporary buffers, and a
``reset`` operation is provided to support that use case.

The ``zeroinit_memory_block`` is just like pod_memory_block,
but initializes the memory it allocates to zero before
returning it. Types have a flag indicating whether they
require zero-initialization or not.

The ``external_memory_block`` is for holding on to data owned
by a system external to DyND. For example, in some cases
DyND can directly map onto the string data of Python's
immutable strings, by using this type of memory block
and flagging the nd::array as immutable.

### Func Source

The ``func`` namespace is where the ``nd::arrfunc`` and
``nd::callable`` objects are defined, which provide
array function abstractions. These have gone through much
evolution, and are still a work in progress.
