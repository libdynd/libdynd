![DyND Logo](docs/logo/dynd_logo_320px.png)

The DyND Library
================

TravisCI: [![Build Status](https://api.travis-ci.org/libdynd/libdynd.svg?branch=master)](https://travis-ci.org/libdynd/libdynd) AppVeyor: [![Build status](https://ci.appveyor.com/api/projects/status/92o89tiw6wwliuxy/branch/master?svg=true)](https://ci.appveyor.com/project/libdynd/libdynd/branch/master)

The core DyND developer team consists of
[Mark Wiebe](https://github.com/mwiebe),
[Irwin Zaid](https://github.com/izaid), and [Ian Henriksen](https://github.com/insertinterestingnamehere). Much of the funding that made this
project possible came through [Continuum Analytics](http://continuum.io/)
and [DARPA-BAA-12-38](https://www.fbo.gov/index?s=opportunity&mode=form&id=7a77846c73ffc5cb22f9295ffe6cdd55&tab=core&_cview=0),
part of [XDATA](http://www.darpa.mil/Our_Work/I2O/Programs/XDATA.aspx).

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
[the Python bindings github site](https://github.com/libdynd/dynd-python).

To get started as a developer of libdynd, begin by working through the
[LibDyND Developer Guide](docs/developer-guide.md). To discuss
the development of this library, subscribe to the
[LibDyND Development List](https://groups.google.com/forum/#!forum/libdynd-dev).

A Brief History Of DyND
=======================

DyND was started in the autumn of 2011 by
[Mark Wiebe](https://github.com/mwiebe), as a private project to begin
dabbling in ideas for how a dynamic multi-dimensional array library
could be structured in C++. During the early formation of
[Continuum Analytics](http://continuum.io/about-continuum), DyND was
open sourced and brought into the company as a part of the
[Blaze project](http://blaze.pydata.org/).

Continuum secured funding for the Blaze project through DARPA's
[XDATA program](http://www.darpa.mil/Our_Work/I2O/Programs/XDATA.aspx),
giving the project space and time to develop as it needed and providing
real data sets and challenge problems to tackle and measure against.

DyND attracted its first major outside contributor,
[Irwin Zaid](https://github.com/izaid), from across the Atlantic during
the cold of winter in 2014. Warmed by the heat of a GPU, Irwin began
by contributing early [CUDA](https://developer.nvidia.com/about-cuda)
support to the library, playing a gradually increasing role in the
design and goals of the project.

DyND is still in an experimental mode, with some mature components and
others severely lacking. Current focus is on the completion of the
[ArrFunc object](docs/arrfuncs.md) to represent array functions and to
flesh out a basic set of functionality modeled after
[NumPy](http://www.numpy.org/). New contributors are welcome, so if you
have the patience to collaborate on a maturing code base, and enjoy C++,
array-oriented, and numeric programming, DyND might be the the open source
project you're looking for.

Building
========

The build system of this library is based on CMake. See
the [build instructions](BUILD.md) for details about how
to build the library.

DyND requires a C++11 compiler, the minimum versions supported are gcc 4.9,
MSVC 2015, and Clang 3.4. The last release to support C++98 was DyND 0.6.6.
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
