[![DyND Logo](docs/logo/dynd_logo_320px.png)](http://libdynd.org)

The DyND Library
================

Travis CI: [![Build Status](https://api.travis-ci.org/libdynd/libdynd.svg?branch=master)](https://travis-ci.org/libdynd/libdynd) AppVeyor: [![Build status](https://ci.appveyor.com/api/projects/status/92o89tiw6wwliuxy/branch/master?svg=true)](https://ci.appveyor.com/project/libdynd/libdynd/branch/master) Coveralls: [![Coverage Status](https://coveralls.io/repos/github/libdynd/libdynd/badge.svg?branch=master)](https://coveralls.io/github/libdynd/libdynd?branch=master)
Gitter: [![Join the chat at https://gitter.im/libdynd/libdynd](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/libdynd/libdynd?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

DyND is a dynamic array library for structured and semi-structured data, written with
C++ as a first-class target and extended to Python with a lightweight binding. It aims
to be a cross-language platform for data analysis, by bringing the popularity and flexibility
of the Python data science stack to other languages. It is inspired by NumPy, the Python
array programming library at the core of the scientific Python stack, but tries to address
a number of obstacles encountered by some of NumPy’s users. Examples of these are support
for variable-sized strings, missing values, variable-sized (ragged) array dimensions, and
versatile tools for creating functions that apply generic patterns across arrays.

At a high level, the cornerstones of DyND are its type system, array container, and callable
(function) objects. These represent the description, storage, and manipulation of dynamic,
reinterpretable bytes across languages. At a low level, DyND defines an primitive execution
kernel that brings together computation and data into a compact form able to execute rapidly
across array elements.

DyND was created by [Irwin Zaid](https://github.com/izaid) and [Mark Wiebe](https://github.com/mwiebe).
The core team consists of [Irwin Zaid](https://github.com/izaid), [Mark Wiebe](https://github.com/mwiebe),
and [Ian Henriksen](https://github.com/insertinterestingnamehere). Others who made important
contributions are [Phillip Cloud](https://github.com/cpcloud), [Michael Droettboom](https://github.com/mdboom),
[Stefan Krah](), [Travis Oliphant](https://en.wikipedia.org/wiki/Travis_Oliphant), and
[Andy Terrel](http://andy.terrel.us/). Much of the funding that made this project possible came through [Continuum Analytics](http://continuum.io/)
and [DARPA-BAA-12-38](https://www.fbo.gov/index?s=opportunity&mode=form&id=7a77846c73ffc5cb22f9295ffe6cdd55&tab=core&_cview=0),
part of [XDATA](http://www.darpa.mil/Our_Work/I2O/Programs/XDATA.aspx). We

We pronounce DyND as "dined", though others refer to it as "dy-n-d". It's not something we're picky about it.

Getting Started
================

This library is actively developed together with its Python
bindings. The Python bindings provide a good way to become familiar
with the library from a high level perspective. See
[the github site for the Python bindings](https://github.com/libdynd/dynd-python).

C++ is a first-class target of the library, the intent is that all
its features should be easily usable in the language. This approach
makes it so that DyND can expose a more uniform interface to C++,
Python, and hopefully other languages that eventually get bindings
to the core DyND library.

DyND is still experimental, so many of the interfaces provided here will continue to change.
That said, feedback and bug reports are greatly appreciated.

To get started as a developer of libdynd, begin by working through the
[LibDyND Developer Guide](docs/developer-guide.md). To discuss
the development of this library, subscribe to the
[LibDyND Development List](https://groups.google.com/forum/#!forum/libdynd-dev).

History Of DyND
=======================

DyND was started in the autumn of 2011 by [Mark Wiebe](https://github.com/mwiebe),
as a private project to begin dabbling in ideas for how a dynamic multidimensional array
library could be structured in C++. During the early formation of [Continuum Analytics](http://continuum.io/about-continuum),
DyND was open sourced and brought into the company as a part of the [Blaze project](http://blaze.pydata.org/).

Continuum secured funding for the Blaze project through DARPA's
[XDATA program](http://www.darpa.mil/Our_Work/I2O/Programs/XDATA.aspx),
giving the project space and time to develop as it needed and providing
real data sets and challenge problems to tackle and measure against.

[Irwin Zaid](https://github.com/izaid) joined the project in the winter of 2014, as
its first major outside contributor. He initially added early [CUDA](https://developer.nvidia.com/about-cuda)
support to the library, but played a gradually increasing role in the design and goals of the project.

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

DyND requires a C++14 compiler, the minimum versions supported are gcc 4.9,
MSVC 2015, and Clang 3.4. The last release to support C++98 was DyND 0.6.6.
An example improvement C++14 brings is a roughly factor of two compile time
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
