DyND Developer Guide
====================

This document is an introduction to DyND for any developers
who would like to understand the inner workings of the code,
and possibly even contribute changes to the DyND codebase.
While many aspects of its design have solidified, many parts of
the code and design are still undergoing large changes, driven
primarily by adding new functionality and connecting DyND to
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
click on the test_dynd project and choose "Set as StartUp Project",
so when you hit play, the tests will run.

If you're building the Python bindings as well, the Python tests can
be run with a command like

    python -c "import dynd;dynd.test()"

Source Code Directory Layout
----------------------------

The main directories to look at within DyND are the 'src' and 'include'
directories. These contain the main source code and header files.

The source files are mostly distributed in subdirectories of 'src/dynd',
with a few that are the main objects or not easily categorized in 'src/dynd'
itself. The same applies to the headers in the 'include/dynd' directory. In
general, if a header .hpp file has a corresponding .cpp file, it goes into
the 'src/*' directory corresponding to 'include/*'.

### Main Source

In the 'include/dynd' directory, the files implementing the main DyND
objects are 'dtype.hpp' and 'ndobject.hpp'. The file 'config.hpp'
has some per-compiler logic to enable/disable various C++11 features,
define some common macros, and declare the version number strings.

### DTypes Source

All the DyND types are presently located in 'include/dynd/dtypes'.
The files 'include/dtype.hpp' basically contain a smart pointer which
wraps instances of type 'base_dtype *' in a more convenient interface.
The file 'include/dtypes/type_id.hpp' contains a number of primitive
enumeration types and some type_id metaprograms.

There are a number of 'base_<kind>_dtype' dtype classes which correspond
to a number of the dtype 'kinds'. For example, if a dtype is of 'uniform_dim'
kind, then it must inherit from 'base_uniform_dim_dtype'.

The code which consumes and produces Blaze datashapes is in
'include/dtypes/datashape_parser.hpp' and 'include/dtypes/datashape_formatter.hpp'.

### Kernels Source

Kernels in DyND are mostly implemented in the 'include/kernels' directory.
All kernels get built in the ckernel_builder class, which is defined in
'include/kernels/ckernel_builders.hpp'. These kernels are intended to be
cheap to construct and execute when they are small, but scale to hold
buffering data and other information when what they are computing is
quite complex.

There are presently three kinds of kernels that are defined and used within
DyND. Assignment kernels, comparison kernels, and element-wise expression
kernels.

Assignment kernels are fundamental to the DyND type system, and
are defined in 'include/kernels/assignment_kernels.hpp'. In some cases,
for example when buffering is required, assignment kernels are used as
child kernels in other types of kernels.

Comparison kernels provide a fixed set of common comparison operations,
including a 'sorting less than' operation which order NaNs and complex
numbers, unlike the regular less than operation. These kernels return
a boolean, and are intended for use by C++ code to implement generic
dynamic algorithms on DyND types.

Element-wise expression kernels are produced by deferred 'expr_dtype'
instances, and is the primary mechanism for DyND to do deferred evaluation.
This system is still fairly immature, but look at
'include/kernels/expr_kernel_generator.hpp', the 'expr_dtype', and the
'unary_expr_dtype' to see its current state. The development driving its
current state is the 'nd.elwise_map' function in the Python exposure.

### Memory Blocks Source

Dynamic ndobject data in DyND is stored in memory blocks, which are
in the 'include/memblock' directory. The base class is defined in
'include/memblock/memory_block.hpp', and the different memory blocks
serve the needs of various dtypes.

The ndobject_memory_block is what ndobjects are. The file
'include/ndobject.hpp' defines an object which contains an
ndobject_memory_block and defines operations on it. One optimization
supported by the ndobject memory block is that data for the
ndobject can be embedded in the same memory as the ndobject.
In this case, the ndobject's data reference is NULL, to avoid
a circular reference.

The fixed_size_pod_memory_block is very simple, for dtypes that are
of a fixed size and alignment.

The pod_memory_block provides a simple memory allocator that doles
out chunks of memory sequentially for variable-sized dtypes. Due to
the way ndobject uses views, similar to NumPy, once some memory has
been allocated within the pod memory block, it cannot be deallocated
until the whole memory block is freed. A small exception is for temporary
buffers, and a 'reset' operation is provided to support that use case.

The zeroinit_memory_block is just like pod_memory_block, but initializes
the memory it allocates to zero before returning it.

The external_memory_block is for holding on to data owned by a system
external to DyND. For example, DyND can directly map onto the string
data of Python's immutable strings, by using this type of memory block
and flagging the ndobject as immutable.

### GFunc Source

GFuncs in DyND provide a mechanism for dynamic function calls, using
DyND ndobjects as the parameter passing mechanism.