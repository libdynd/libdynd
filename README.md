[![DyND Logo](docs/logo/dynd_logo_320px.png)](http://libdynd.org)

The DyND Library
================

Travis CI: [![Build Status](https://api.travis-ci.org/libdynd/libdynd.svg?branch=master)](https://travis-ci.org/libdynd/libdynd) AppVeyor: [![Build status](https://ci.appveyor.com/api/projects/status/92o89tiw6wwliuxy/branch/master?svg=true)](https://ci.appveyor.com/project/libdynd/libdynd/branch/master) Coveralls: [![Coverage Status](https://coveralls.io/repos/github/libdynd/libdynd/badge.svg?branch=master)](https://coveralls.io/github/libdynd/libdynd?branch=master)
Gitter: [![Join the chat at https://gitter.im/libdynd/libdynd](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/libdynd/libdynd?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

DyND is a dynamic array library for structured and semi-structured data, written with
C++ as a first-class target and extended to Python with a lightweight binding. It aims
to be a cross-language platform for data analysis, by bringing the popularity and flexibility
of the Python data science stack to other languages. It is inspired by [NumPy](http://www.numpy.org/),
the Python array programming library at the core of the scientific Python stack, but tries
to address a number of obstacles encountered by some of NumPy’s users. Examples of these are
support for variable-sized strings, missing values, variable-sized (ragged) array dimensions,
and versatile tools for creating functions that apply generic patterns across arrays.

At a high level, the cornerstones of DyND are its type system, array container, and callable
(function) objects. These represent the description, storage, and manipulation of dynamic,
reinterpretable bytes across languages. At a low level, DyND defines an primitive execution
kernel that brings together computation and data in a compact form that can be executed rapidly
across array elements. Where DyND begins to shine is in its support for functional composition.
For example, NumPy-like broadcasting is not built into every callable, rather it is made
available as a functional transformation applied to a scalar callable.

DyND was created by [Irwin Zaid](https://github.com/izaid) and [Mark Wiebe](https://github.com/mwiebe).
The core team consists of [Irwin Zaid](https://github.com/izaid), [Mark Wiebe](https://github.com/mwiebe),
and [Ian Henriksen](https://github.com/insertinterestingnamehere). Others who made important
contributions include [Phillip Cloud](https://github.com/cpcloud), [Michael Droettboom](https://github.com/mdboom),
[Stefan Krah](https://github.com/skrah), [Travis Oliphant](https://en.wikipedia.org/wiki/Travis_Oliphant), and
[Andy Terrel](http://andy.terrel.us/). Much of the funding that made this project possible came through [Continuum Analytics](http://continuum.io/)
and [DARPA-BAA-12-38](https://www.fbo.gov/index?s=opportunity&mode=form&id=7a77846c73ffc5cb22f9295ffe6cdd55&tab=core&_cview=0),
part of [XDATA](http://www.darpa.mil/Our_Work/I2O/Programs/XDATA.aspx).

We pronounce DyND as "dined", though others refer to it as "dy-n-d". It's not something we're picky about it.

Getting Started
===============

This library is actively developed together with its Python
bindings. The Python bindings provide a good way to become familiar
with the library from a high level perspective. See
[the github site for the Python bindings](https://github.com/libdynd/dynd-python).

C++ is a first-class target of the library, the intent is that all features should
be easily usable in that language. This approach makes it so that DyND can expose
a more uniform interface to C++, Python, and hopefully other languages that eventually
get bindings to the core DyND library.

DyND is still experimental, so many of the interfaces provided here will continue to change.
That said, feedback and bug reports are greatly appreciated.

To get started as a developer of libdynd, begin by working through the
[LibDyND Developer Guide](docs/developer-guide.md). To discuss
the development of this library, subscribe to the
[LibDyND Development List](https://groups.google.com/forum/#!forum/libdynd-dev).

History Of DyND
===============

DyND was started as a personal project of [Mark Wiebe](https://github.com/mwiebe)
in September 2011 to begin dabbling in ideas for how a dynamic, multidimensional
array library could be structured in C++. See [here](https://github.com/libdynd/libdynd/commit/768ac9a30cdb4619d09f4656bfd895ab2b91185d)
for the very first commit. Mark was at the [University of British Columbia](https://www.ubc.ca/),
then joined [Continuum Analytics](http://continuum.io/about-continuum) part-time when
it was founded in January 2012, and later became full-time in the spring of 2012. He parted ways with Continuum at the end of 2014, joining [Thinkbox Software](http://www.thinkboxsoftware.com).

During the formation of Continuum, DyND was open-sourced and brought into the company
as a part of the [Blaze project](http://blaze.pydata.org). Continuum secured funding
for the Blaze project through DARPA's [XDATA program](http://www.darpa.mil/Our_Work/I2O/Programs/XDATA.aspx),
giving the project space and time to develop as it needed and providing real data sets
and challenge problems to tackle and measure against. [Travis Oliphant](https://en.wikipedia.org/wiki/Travis_Oliphant)
has told the story of the early days at Continuum, and why it supports DyND, on the
NumPy mailing list - see [here](https://mail.scipy.org/pipermail/numpy-discussion/2015-August/073412.html).

[Irwin Zaid](https://github.com/izaid) joined the project in the winter of 2014, as
its first major outside contributor, while he was a research fellow at [Christ Church](http://www.chch.ox.ac.uk),
[University of Oxford](http://www.ox.ac.uk). He initially added [CUDA](https://developer.nvidia.com/about-cuda)
support to the library, then played a gradually increasing role in the design and goals of the project.
He took over development of DyND in the spring of 2015, and was funded by Continuum from June 2015
until April 2016.

[Ian Henriksen](https://github.com/insertinterestingnamehere) began working on DyND through
[Google Summer of Code](https://developers.google.com/open-source/gsoc/) - under the umbrella
of [NumFocus](http://www.numfocus.org) - in the summer of 2015. He remained active in the
project from then onwards.

DyND is still experimental. Some of its components are mature while others are severely lacking.
New contributors are welcome, so if you have the patience to collaborate on a maturing code base, and enjoy C++,
array-oriented, and numeric programming, DyND might be the the open source project you're looking for.

Building
========

The build system of this library is based on CMake. See
the [build instructions](BUILD.md) for details about how
to build the library.

DyND requires a C++14 compiler, the minimum versions supported are gcc 4.9,
MSVC 2015, and Clang 3.4. The last release to support C++98 was DyND 0.6.6.
C++14 brings several things to the library, including a roughly factor of two
compile time improvement and generality with variadic templates.

Documentation
=============

[Documentation Index](docs/index.md)

Running The Tests
=================

The tests are built using Google Test. To execute the test suite,
run the `test_libdynd` program.
