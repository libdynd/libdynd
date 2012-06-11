cmake_minimum_required(VERSION 2.8)
project(dnd_tests)

set(tests_SRC
    test_array_dtype.cpp
    test_byteswap_dtype.cpp
    test_conversion_dtype.cpp
    test_arithmetic_op.cpp
    test_dtype.cpp
    test_dtype_assign.cpp
    test_dtype_casting.cpp
    test_dtype_promotion.cpp
    test_ndarray.cpp
    test_ndarray_assign.cpp
    test_ndarray_index.cpp
    test_ndarray_arange.cpp
    test_shape_tools.cpp
    )

# TODO: this is awkwardly hardcoded
if(WIN32)
include_directories(
    ../../../googletest/include
    ../include
    .
    )
else(WIN32)
include_directories(
    ../../AnimPhys/gtest/include
    ../include
    .
    )
endif(WIN32)

add_executable(dnd_tests ${tests_SRC})

# gtest generates "unnamed-type-template-args" warnings with clang
if(WIN32)
    # TODO: Make this not be hardcoded
    target_link_libraries(dnd_tests
        dynamicndarray
        ../../../googletest/build-vc10/RelWithDebInfo/gtest
        ../../../googletest/build-vc10/RelWithDebInfo/gtest_main
        )
else(WIN32)
    set_target_properties(dnd_tests
                PROPERTIES COMPILE_FLAGS
                "-Wno-unnamed-type-template-args -pthread")

    target_link_libraries(dnd_tests
        dynamicndarray
        pthread
        /home/mwiebe/Develop/AnimPhys/gtest/make/gtest-all.o
        /home/mwiebe/Develop/AnimPhys/gtest/make/gtest_main.o
        )
endif(WIN32)


# Compile-time tests: test code that is supposed to produce compile errors
# I couldn't find a properly specified way to do this, so hacked together
# this function.
#set(TEST_BUILDERROR_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../include)
#include(TestExpectedCompileError.cmake)

#test_expected_compile_succeed(builderror_includendarray.cpp)
#test_expected_compile_error(builderror_badassignment.cpp)
