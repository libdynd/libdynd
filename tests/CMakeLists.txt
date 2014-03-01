#
# Copyright (C) 2011-14 Mark Wiebe, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(test_libdynd)

# Disable optimizations so the tests build quicker
if(WIN32)
    # _VARIADIC_MAX=10 is for VS2012
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_VARIADIC_MAX=10 /Od")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_USE_OWN_TR1_TUPLE=1 -O0")
endif()

set(tests_SRC
    codegen/test_codegen_cache.cpp
    codegen/test_unary_kernel_adapter.cpp
    codegen/test_binary_kernel_adapter.cpp
#    codegen/assembly_samples/asm_tests.cpp
    types/test_bytes_type.cpp
    types/test_byteswap_type.cpp
    types/test_categorical_type.cpp
    types/test_complex_type.cpp
    types/test_convert_type.cpp
    types/test_align_type.cpp
    types/test_char_type.cpp
    types/test_cuda_host_type.cpp
    types/test_cuda_device_type.cpp
    types/test_datashape_formatter.cpp
    types/test_datashape_parser.cpp
    types/test_date_type.cpp
    types/test_datetime_type.cpp
    types/test_type.cpp
    types/test_type_type.cpp
    types/test_type_assign.cpp
    types/test_type_casting.cpp
    types/test_type_promotion.cpp
    types/test_fixedbytes_type.cpp
    types/test_fixed_dim_type.cpp
    types/test_fixedstring_type.cpp
    types/test_cstruct_type.cpp
    types/test_groupby_type.cpp
    types/test_json_type.cpp
    types/test_pointer_type.cpp
    types/test_strided_dim_type.cpp
    types/test_string_type.cpp
    types/test_struct_type.cpp
    types/test_tuple_type.cpp
    types/test_var_dim_type.cpp
    gfunc/test_callable.cpp
    gfunc/test_ckernel_deferred.cpp
    array/test_json_formatter.cpp
    array/test_json_parser.cpp
    array/test_array.cpp
    array/test_array_range.cpp
    array/test_array_assign.cpp
    array/test_array_at.cpp
    array/test_array_cast.cpp
    array/test_array_compare.cpp
    array/test_array_views.cpp
	array/test_memmap.cpp
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
source_group("Types Tests" REGULAR_EXPRESSION "\\/types\\/test_[^\\/]*\\.cpp$")
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
