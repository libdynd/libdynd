Dynamic NDArray Library
=======================

This is a C++ library for dynamic, multidimensional arrays, inspired by
the success of Numpy and based on ideas for how it could be better. It
is in a very preliminary state, so every aspect of its API and interface
could likely change drastically while learning how it should look by
trying to use it in practical settings.

This library includes exposure to Python 2.7, implemented partially in
Cython. This support is interoperable with Numpy, allowing the ndarray
library to focus on things Numpy isn't good at, and letting users
work with both systems as seamlessly as possible.

Building and Installing
=======================

The build system of this library is based on CMake. See INSTALL.txt for
details about how to build and install the library.

Running The Tests
=================

There are two sets of tests, C++ tests built using Google Tests, and
Python Tests which can be run with Nose.

To run the Python, tests, switch into the dynamicndarray/python/pyddnd/tests
directory, and run the command "nosetests ."
