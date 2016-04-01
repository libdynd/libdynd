Running C++ Tests
=================

The project in the `tests` subfolder contains a testsuite
using google test. To run it, simply execute the `test_libdynd`
executable. Here's how that looks on Windows:


    D:\Develop\dynd\build\tests\RelWithDebInfo>test_dynd.exe
    Running main() from gtest_main.cc
    [==========] Running 244 tests from 36 test cases.
    [----------] Global test environment set-up.
    [----------] 2 tests from BytesDType
    [ RUN      ] BytesDType.Create
    [       OK ] BytesDType.Create (0 ms)
    [ RUN      ] BytesDType.Assign
    [       OK ] BytesDType.Assign (0 ms)
    [----------] 2 tests from BytesDType (2 ms total)

    [----------] 3 tests from ByteswapDType
    [ RUN      ] ByteswapDType.Create

    <snip>

    [       OK ] ShapeTools.MultiStridesToAxisPerm_TwoOps (0 ms)
    [----------] 3 tests from ShapeTools (1 ms total)

    [----------] Global test environment tear-down
    [==========] 244 tests from 36 test cases ran. (152 ms total)
    [  PASSED  ] 244 tests.

If you want to see what options there are for running tests,
run `test_dynd --help`. One useful ability is to filter tests
with a white or black list.

    D:\Develop\dynd\build\tests\RelWithDebInfo>test_libdynd.exe --gtest_filter=DType.*
    Running main() from gtest_main.cc
    Note: Google Test filter = DType.*
    [==========] Running 1 test from 1 test case.
    [----------] Global test environment set-up.
    [----------] 1 test from DType
    [ RUN      ] DType.BasicConstructor
    [       OK ] DType.BasicConstructor (1 ms)
    [----------] 1 test from DType (1 ms total)

    [----------] Global test environment tear-down
    [==========] 1 test from 1 test case ran. (4 ms total)
    [  PASSED  ] 1 test.

One useful option is to disable googletest's catching of exceptions,
with the option `--gtest_catch_exceptions=0`. This allows your debugger
to handle it.

To generate Jenkins-compatible XML output, use `test_dynd --gtest_output=xml:test_dynd_results.xml`.


GETTING TEST COVERAGE REPORTS
=============================

Code coverage reports for the unit test suite can be generated.  The
full library must be recompiled with `gcov` flags turned on, and
[lcov](http://ltp.sourceforge.net/coverage/lcov.php) must be installed
to generate the report.

The easiest way to do this is to make a separate CMake build directory
with the `DYND_COVERAGE` parameter set to on::

  ```
  ~ $ cd libdynd
  ~/libdynd $ mkdir build-coverage
  ~/libdynd $ cd build-coverage
  ~/libdynd/build $ cmake -DDYND_COVERAGE=ON ..
  <...>
  ~/libdynd/build $ make
  <...>
  ~/libdynd/build $ make coverage
  ```

View the resulting `coverage/index.html` in your web browser to see
the code coverage.
