#
# Copyright (C) 2011-13 Mark Wiebe, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(test_libdynd)

# Disable optimizations so the tests build quicker
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_USE_OWN_TR1_TUPLE=1 -O0")
endif()

set(tests_SRC
    codegen/test_codegen_cache.cpp
    codegen/test_unary_kernel_adapter.cpp
    codegen/test_binary_kernel_adapter.cpp
#    codegen/assembly_samples/asm_tests.cpp
    dtypes/test_bytes_dtype.cpp
    dtypes/test_byteswap_dtype.cpp
    dtypes/test_categorical_dtype.cpp
    dtypes/test_complex_dtype.cpp
    dtypes/test_convert_dtype.cpp
    dtypes/test_align_dtype.cpp
    dtypes/test_datashape_formatter.cpp
    dtypes/test_datashape_parser.cpp
    dtypes/test_date_dtype.cpp
    dtypes/test_datetime_dtype.cpp
    dtypes/test_dtype.cpp
    dtypes/test_dtype_dtype.cpp
    dtypes/test_dtype_assign.cpp
    dtypes/test_dtype_casting.cpp
    dtypes/test_dtype_promotion.cpp
    dtypes/test_fixedbytes_dtype.cpp
    dtypes/test_fixed_dim_type.cpp
    dtypes/test_fixedstring_type.cpp
    dtypes/test_cstruct_type.cpp
    dtypes/test_groupby_dtype.cpp
    dtypes/test_json_dtype.cpp
    dtypes/test_pointer_dtype.cpp
    dtypes/test_strided_dim_type.cpp
    dtypes/test_string_type.cpp
    dtypes/test_struct_type.cpp
    dtypes/test_tuple_dtype.cpp
    dtypes/test_var_dim_type.cpp
    gfunc/test_callable.cpp
    array/test_json_formatter.cpp
    array/test_json_parser.cpp
    array/test_array.cpp
    array/test_array_range.cpp
    array/test_array_assign.cpp
    array/test_array_at.cpp
    array/test_array_cast.cpp
    array/test_array_compare.cpp
    array/test_array_views.cpp
    vm/test_elwise_program.cpp
    test_arithmetic_op.cpp
    test_shape_tools.cpp
    test_platform.cpp
    ../thirdparty/gtest/gtest-all.cc
    ../thirdparty/gtest/gtest_main.cc
    )

include_directories(
    ../thirdparty
    .
    )

source_group("Other Tests" REGULAR_EXPRESSION "\\/test_[^\\/]*\\.cpp$")
source_group("Array Tests" REGULAR_EXPRESSION "\\/array\\/test_[^\\/]*\\.cpp$")
source_group("CodeGen Tests" REGULAR_EXPRESSION "\\/codegen\\/test_[^\\/]*\\.cpp$")
source_group("DTypes Tests" REGULAR_EXPRESSION "\\/dtypes\\/test_[^\\/]*\\.cpp$")
source_group("GFunc Tests" REGULAR_EXPRESSION "\\/gfunc\\/test_[^\\/]*\\.cpp$")
source_group("VM Tests" REGULAR_EXPRESSION "\\/vm\\/test_[^\\/]*\\.cpp$")

add_executable(test_libdynd ${tests_SRC})

if(WIN32)
    target_link_libraries(test_libdynd
        libdynd
        )
elseif(APPLE)
    target_link_libraries(test_libdynd
        libdynd
        )
    if(USE_RELATIVE_RPATH)
        add_custom_command(TARGET test_libdynd
            POST_BUILD COMMAND ${CMAKE_INSTALL_NAME_TOOL}
                -add_rpath "@loader_path/../"
                $<TARGET_FILE:test_libdynd>)
    endif()
else()
    set_target_properties(test_libdynd PROPERTIES
        COMPILE_FLAGS "-pthread")
    #-Wno-unnamed-type-template-args 

    target_link_libraries(test_libdynd
        libdynd
        pthread
        )
endif()

# If installation is requested, install the program
if (DYND_INSTALL_LIB)
    install(TARGETS test_libdynd
        RUNTIME DESTINATION bin)
endif()

# Compile-time tests: test code that is supposed to produce compile errors
# I couldn't find a properly specified way to do this, so hacked together
# this function.
#set(TEST_BUILDERROR_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../include)
#include(TestExpectedCompileError.cmake)

#test_expected_compile_succeed(builderror_includendarray.cpp)
#test_expected_compile_error(builderror_badassignment.cpp)
