#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(dnd_tests)

set(tests_SRC
    codegen/test_codegen_cache.cpp
    codegen/test_unary_kernel_adapter.cpp
    codegen/test_binary_kernel_adapter.cpp
    dtypes/test_array_dtype.cpp
    dtypes/test_byteswap_dtype.cpp
    dtypes/test_categorical_dtype.cpp
    dtypes/test_convert_dtype.cpp
    dtypes/test_align_dtype.cpp
    dtypes/test_dtype.cpp
    dtypes/test_dtype_assign.cpp
    dtypes/test_dtype_casting.cpp
    dtypes/test_dtype_promotion.cpp
    dtypes/test_fixedstring_dtype.cpp
    dtypes/test_string_dtype.cpp
    dtypes/test_tuple_dtype.cpp
    dtypes/test_pointer_dtype.cpp
    ndarray/test_ndarray.cpp
    ndarray/test_ndarray_assign.cpp
    ndarray/test_ndarray_index.cpp
    ndarray/test_ndarray_arange.cpp
    ndarray/test_ndarray_views.cpp
    test_arithmetic_op.cpp
    test_shape_tools.cpp
    test_unary_kernel.cpp
    ../thirdparty/gtest/gtest-all.cc
    ../thirdparty/gtest/gtest_main.cc
    )

include_directories(
    ../thirdparty
    .
    )

source_group("Other Tests" REGULAR_EXPRESSION "\\/test_[^\\/]*\\.cpp$")
source_group("NDArray Tests" REGULAR_EXPRESSION "\\/ndarray\\/test_[^\\/]*\\.cpp$")
source_group("CodeGen Tests" REGULAR_EXPRESSION "\\/codegen\\/test_[^\\/]*\\.cpp$")
source_group("DTypes Tests" REGULAR_EXPRESSION "\\/dtypes\\/test_[^\\/]*\\.cpp$")

add_executable(dnd_tests ${tests_SRC})

# gtest generates "unnamed-type-template-args" warnings with clang
if(WIN32)
    # TODO: Make this not be hardcoded
    target_link_libraries(dnd_tests
        dynamicndarray
        )
elseif(APPLE)
    target_link_libraries(dnd_tests
        dynamicndarray
        )
else()
    set_target_properties(dnd_tests
                PROPERTIES COMPILE_FLAGS
                "-Wno-unnamed-type-template-args -pthread")

    target_link_libraries(dnd_tests
        dynamicndarray
        pthread
        )
endif()


# Compile-time tests: test code that is supposed to produce compile errors
# I couldn't find a properly specified way to do this, so hacked together
# this function.
#set(TEST_BUILDERROR_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../include)
#include(TestExpectedCompileError.cmake)

#test_expected_compile_succeed(builderror_includendarray.cpp)
#test_expected_compile_error(builderror_badassignment.cpp)
