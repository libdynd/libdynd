The DyND Library
================

LibDyND, a component of [the Blaze project](http://blaze.pydata.org/),
is a C++ library for dynamic, multidimensional arrays. It is inspired
by NumPy, the Python array programming library at the core of the
scientific Python stack, but tries to address a number of obstacles
encountered by some of its users. Examples of this are support for
variable-sized string and ragged array types. The library is in a
preview development state, and can be thought of as a sandbox where
features are being tried and tweaked to gain experience with them.

C++ is a first-class target of the library, the intent is that all
its features should be easily usable in the language. This has many
benefits, such as that development within LibDyND using its own
components is more natural than in a library designed primarily
for embedding in another language.

This library is being actively developed together with its Python
bindings, which are a good way to get a taste of the library from
a high level perspective. See
[the Python bindings github site](https://github.com/ContinuumIO/dynd-python).

To discuss the development of this library, subscribe to the
[LibDyND Development List](https://groups.google.com/forum/#!forum/libdynd-dev).

Building
========

The build system of this library is based on CMake. See
the [build instructions](BUILD.md) for details about how
to build the library.

Documentation
=============

[Documentation Index](documents/index.md)

Running The Tests
=================

The tests are built using Google Test. To execute the test suite,
run the `test_dynd` program.
