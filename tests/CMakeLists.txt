#
# Copyright (C) 2011-15 DyND Developers
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
    dynd_assertions.hpp
    pp/test_pp_arithmetic.cpp
    pp/test_pp_comparison.cpp
    pp/test_pp_if.cpp
    pp/test_pp_list.cpp
    pp/test_pp_logical.cpp
    pp/test_pp_token.cpp
    codegen/test_codegen_cache.cpp
    codegen/test_unary_kernel_adapter.cpp
    codegen/test_binary_kernel_adapter.cpp
#    codegen/assembly_samples/asm_tests.cpp
    types/test_align_type.cpp
    types/test_bytes_type.cpp
    types/test_byteswap_type.cpp
    types/test_categorical_type.cpp
    types/test_complex_type.cpp
    types/test_convert_type.cpp
    types/test_cfixed_dim_type.cpp
    types/test_char_type.cpp
    types/test_cstruct_type.cpp
    types/test_ctuple_type.cpp
    types/test_cuda_host_type.cpp
    types/test_cuda_device_type.cpp
    types/test_datashape_formatter.cpp
    types/test_datashape_parser.cpp
    types/test_date_type.cpp
    types/test_datetime_type.cpp
    types/test_fixed_dim_type.cpp
    types/test_fixedbytes_type.cpp
    types/test_fixedstring_type.cpp
    types/test_groupby_type.cpp
    types/test_json_type.cpp
    types/test_option_type.cpp
    types/test_pointer_type.cpp
    types/test_string_type.cpp
    types/test_struct_type.cpp
    types/test_symbolic_types.cpp
    types/test_time_type.cpp
    types/test_tuple_type.cpp
    types/test_type.cpp
    types/test_type_type.cpp
    types/test_type_assign.cpp
    types/test_type_casting.cpp
    types/test_type_substitute.cpp
    types/test_type_pattern_match.cpp
    types/test_type_promotion.cpp
    types/test_var_dim_type.cpp
    func/special_vals.hpp
    func/test_apply.cpp
    func/test_arithmetic.cpp
    func/test_arrfunc.cpp
    func/test_callable.cpp
    func/test_chain_arrfunc.cpp
    func/test_elwise.cpp
    func/test_fft.cpp
    func/test_functor_arrfunc.cpp
    func/test_neighborhood.cpp
    func/test_multidispatch.cpp
    func/test_permute.cpp
    func/test_random.cpp
    func/test_reduction.cpp
    func/test_registry.cpp
    func/test_rolling.cpp
    func/test_special.cpp
    func/test_take.cpp
    func/test_take_by_pointer.cpp
    array/test_array.cpp
    array/test_array_range.cpp
    array/test_array_assign.cpp
    array/test_array_at.cpp
    array/test_array_cast.cpp
    array/test_array_compare.cpp
    array/test_array_views.cpp
    array/test_arrmeta_holder.cpp
    array/test_json_formatter.cpp
    array/test_json_parser.cpp
    array/test_memmap.cpp
    array/test_view.cpp
    vm/test_elwise_program.cpp
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
source_group("Func Tests" REGULAR_EXPRESSION "\\/func\\/test_[^\\/]*\\.cpp$")
source_group("VM Tests" REGULAR_EXPRESSION "\\/vm\\/test_[^\\/]*\\.cpp$")

if (DYND_CUDA)
    # Replace some source files with their CUDA versions
    list(REMOVE_ITEM tests_SRC
        func/test_apply.cpp
        func/test_elwise.cpp
        )
    list(APPEND tests_SRC
        func/test_apply.cu
        func/test_elwise.cu
        )
endif()

if (DYND_CUDA)
    cuda_add_executable(test_libdynd ${tests_SRC} OPTIONS -ldynddev -L${CMAKE_BINARY_DIR})
    set(LINK_LIBS ${LINK_LIBS} libdynddev)
else()
    add_executable(test_libdynd ${tests_SRC})
    set(LINK_LIBS ${LINK_LIBS} libdynd)
endif()

if(WIN32)
    target_link_libraries(test_libdynd
        ${LINK_LIBS}
        )
elseif(APPLE)
    target_link_libraries(test_libdynd
        ${LINK_LIBS}
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
        ${LINK_LIBS}
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
