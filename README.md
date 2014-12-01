The DyND Library
================

TravisCI: [![Build Status](https://api.travis-ci.org/ContinuumIO/libdynd.svg?branch=master)](https://travis-ci.org/ContinuumIO/libdynd) AppVeyor: [![Build Status](https://ci.appveyor.com/api/projects/status/fbds4gsjnoxw5m96/branch/master?svg=true)](https://ci.appveyor.com/project/mwiebe/libdynd)

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

DyND requires a C++11 compiler, the minimum versions supported are gcc 4.7,
MSVC 2013 Update 4, and Clang 3.4. The last release to support C++98 was DyND 0.6.6.
An example improvement C++11 brings is a roughly factor of two compile time
improvement and increased generality by using variadic templates instead of
preprocessor metaprogramming. Many excellent projects such as [LLVM](http://llvm.org/)
and [libelemental](http://libelemental.org/) have already adopted the newer
standard, it makes a lot of sense for dynd to embrace it as it matures.

Documentation
=============

[Documentation Index](docs/index.md)

Running The Tests
=================

The tests are built using Google Test. To execute the test suite,
run the `test_dynd` program.
