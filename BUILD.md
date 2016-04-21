PREREQUISITES
=============

If you want to contribute to DyND development, you should use the combined
Python/C++ build [described here](https://github.com/libdynd/dynd-python/blob/master/BUILD_INSTALL.md).
In this configuration, it is easier to ensure that both libdynd and its Python
bindings always build and pass their tests.

This library requires a C++14 compiler. On Windows,
Visual Studio 2015 is the minimum supported compiler.
Clang 3.4 or gcc 4.9 or newer is recommended on other platforms.

 * CMake >= 2.8.11

The following libraries/projects are included with the code:

 * Google Test 1.6 (included in project)

BUILD/INSTALL INSTRUCTIONS
==========================

CMake is the only supported build system for this library. This
may expand in the future, but for the time being this is the
only one which will be kept up to date.

Linux and OS X
--------------

  ```
  ~ $ git clone --recursive https://github.com/libdynd/libdynd.git
  <...>
  ~ $ cd libdynd
  ~/libdynd $ mkdir build
  ~/libdynd $ cd build
  ~/libdynd/build $ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
  <...>
  ~/libdynd/build $ make
  <...>
  ~/libdynd/build $ sudo make install
  ```

Windows
-------

  ```
  C:\> git clone --recursive https://github.com/libdynd/libdynd.git
  <...>
  C:\> cd libdynd
  C:\libdynd>mkdir build
  C:\libdynd>cd build
  C:\libdynd\build>cmake -G "Visual Studio 14 2015 Win64" ..
  <...>
  C:\libdynd\build>start libdynd.sln
  ```

Select the desired build type, for example RelWithDebInfo, then build. You may want
to right click on `test_libdynd` and set it as the startup project so you can run the
tests by hitting Control-F5.
