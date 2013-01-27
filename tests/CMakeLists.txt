#
# Copyright (C) 2011-13, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(test_dynd)

set(tests_SRC
    codegen/test_codegen_cache.cpp
    codegen/test_unary_kernel_adapter.cpp
    codegen/test_binary_kernel_adapter.cpp
#    codegen/assembly_samples/asm_tests.cpp
    dtypes/test_bytes_dtype.cpp
    dtypes/test_byteswap_dtype.cpp
    dtypes/test_categorical_dtype.cpp
    dtypes/test_convert_dtype.cpp
    dtypes/test_align_dtype.cpp
    dtypes/test_datashape_parser.cpp
    dtypes/test_date_dtype.cpp
    dtypes/test_dtype.cpp
    dtypes/test_dtype_assign.cpp
    dtypes/test_dtype_casting.cpp
    dtypes/test_dtype_promotion.cpp
    dtypes/test_fixedarray_dtype.cpp
    dtypes/test_fixedstring_dtype.cpp
    dtypes/test_fixedstruct_dtype.cpp
    dtypes/test_json_dtype.cpp
    dtypes/test_pointer_dtype.cpp
    dtypes/test_strided_array_dtype.cpp
    dtypes/test_string_dtype.cpp
    dtypes/test_struct_dtype.cpp
    dtypes/test_tuple_dtype.cpp
    dtypes/test_var_array_dtype.cpp
    gfunc/test_callable.cpp
    ndobject/test_json_parser.cpp
    ndobject/test_ndobject.cpp
    ndobject/test_ndobject_arange.cpp
    ndobject/test_ndobject_assign.cpp
    ndobject/test_ndobject_at.cpp
    ndobject/test_ndobject_views.cpp
    vm/test_elwise_program.cpp
    test_arithmetic_op.cpp
    test_shape_tools.cpp
    ../thirdparty/gtest/gtest-all.cc
    ../thirdparty/gtest/gtest_main.cc
    )

include_directories(
    ../thirdparty
    .
    )

source_group("Other Tests" REGULAR_EXPRESSION "\\/test_[^\\/]*\\.cpp$")
source_group("NDArray Tests" REGULAR_EXPRESSION "\\/ndarray\\/test_[^\\/]*\\.cpp$")
source_group("NDObject Tests" REGULAR_EXPRESSION "\\/ndobject\\/test_[^\\/]*\\.cpp$")
source_group("CodeGen Tests" REGULAR_EXPRESSION "\\/codegen\\/test_[^\\/]*\\.cpp$")
source_group("DTypes Tests" REGULAR_EXPRESSION "\\/dtypes\\/test_[^\\/]*\\.cpp$")
source_group("GFunc Tests" REGULAR_EXPRESSION "\\/gfunc\\/test_[^\\/]*\\.cpp$")
source_group("VM Tests" REGULAR_EXPRESSION "\\/vm\\/test_[^\\/]*\\.cpp$")

add_executable(test_dynd ${tests_SRC})

if(WIN32)
    target_link_libraries(test_dynd
        dynd
        )
elseif(APPLE)
    target_link_libraries(test_dynd
        dynd
        )
else()
    set_target_properties(test_dynd
                PROPERTIES COMPILE_FLAGS
                "-Wno-unnamed-type-template-args -pthread")

    target_link_libraries(test_dynd
        dynd
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
