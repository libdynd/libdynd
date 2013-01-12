PREREQUISITES
=============

This library requires a C++98 or C++11 compiler. On Windows, Visual
Studio 2010 is the recommended compiler. On Mac OS X, clang
is the recommended compiler. On Linux, gcc 4.6.1 and 4.7.0 have
been tested.

 * CMake >= 2.6

The following libraries/projects are included with the code:

 * Google Test 1.6 (included in project)

BUILD INSTRUCTIONS
==================

CMake is the only supported build system for this library. This
may expand in the future, but for the time being this is the
only one which will be kept up to date.

Windows
-------
Visual Studio 2010 or newer is recommended.

1. Run CMake-gui.

2. For the 'source code' folder, choose the
    dynd folder which is the root of the project.

3. For the 'build the binaries' folder, create a 'build'
    subdirectory so that your build is isolated from the
    source code files.

4. Double-click on the generated dynd.sln
    to open Visual Studio. The RelWithDebInfo configuration is
    recommended for most purposes.

*OR*

Start a command prompt window, and navigate to the
dynd folder which is the root of the project.
Switch the "-G" argument below to "Visual Studio 10" if using
32-bit Python.
Execute the following commands:

    D:\dynd>mkdir build
    D:\dynd>cd build
    D:\dynd\build>cmake -G "Visual Studio 10 Win64" ..
       [output, check it for errors]
    D:\dynd\build>start dynd.sln
       [Visual Studio should start and load the project]

The RelWithDebInfo configuration is recommended for most purposes.

Linux
-----

Execute the following commands from the dynd folder,
which is the root of the project:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

Mac OS X
--------

Switch the "-DCMAKE\_OSX\_ARCHITECTURES" argument below to "i386" if
you're using 32-bit Python. Execute the following commands
from the dynd folder, which is the root of the project:

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_FLAGS="-stdlib=libc++"  ..
    $ make

