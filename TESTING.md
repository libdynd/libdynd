Running C++ Tests
=================

The project in the `tests` subfolder contains a testsuite
using google test. To run it, simply execute the `test_dynd`
executable. Here's how that looks on Windows:


    D:\Develop\dynamicndarray\build\tests\RelWithDebInfo>test_dynd.exe
    Running main() from gtest_main.cc
    [==========] Running 83 tests from 23 test cases.
    [----------] Global test environment set-up.
    [----------] 1 test from CodeGenCache
    [ RUN      ] CodeGenCache.UnaryCaching
    [       OK ] CodeGenCache.UnaryCaching (0 ms)
    [----------] 1 test from CodeGenCache (1 ms total)

    [----------] 2 tests from UnaryKernelAdapter
    [ RUN      ] UnaryKernelAdapter.BasicOperations

    <snip>

    [       OK ] UnaryKernel.Specialization (0 ms)
    [----------] 1 test from UnaryKernel (0 ms total)

    [----------] Global test environment tear-down
    [==========] 83 tests from 23 test cases ran. (287 ms total)
    [  PASSED  ] 83 tests.

If you want to see what options there are for running tests,
run `test_dynd --help`. One useful ability is to filter tests
with a white or black list.

    D:\Develop\dynamicndarray\build\tests\RelWithDebInfo>test_dynd.exe --gtest_filter=DType.*
    Running main() from gtest_main.cc
    Note: Google Test filter = DType.*
    [==========] Running 2 tests from 1 test case.
    [----------] Global test environment set-up.
    [----------] 2 tests from DType
    [ RUN      ] DType.BasicConstructor
    [       OK ] DType.BasicConstructor (0 ms)
    [ RUN      ] DType.SingleCompare
    [       OK ] DType.SingleCompare (1 ms)
    [----------] 2 tests from DType (2 ms total)

    [----------] Global test environment tear-down
    [==========] 2 tests from 1 test case ran. (6 ms total)
    [  PASSED  ] 2 tests.

One useful option is to disable googletest's catching of exceptions,
with the option `--gtest_catch_exceptions=0`. This allows your debugger
to handle it.

To generate Jenkins-compatible XML output, use `test_dynd --gtest_output=xml:test_dynd_results.xml`.
