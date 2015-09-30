#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(test_libdynd)

# Disable optimizations so the tests build quicker
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # _VARIADIC_MAX=10 is for VS2012
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_VARIADIC_MAX=10 /Od")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # The gtest macro GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_
    # doesn't work to suppress unreachable code warnings with recent versions
    # of clang, so disable that warning for the tests so they compile properly.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_USE_OWN_TR1_TUPLE=1 -Wno-unused-value -Wno-ignored-attributes")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_USE_OWN_TR1_TUPLE=1")
endif()

set(tests_SRC
    dynd_assertions.hpp
    types/test_align_type.cpp
    types/test_array_type.cpp
    types/test_bytes_type.cpp
    types/test_byteswap_type.cpp
    types/test_c_contiguous_type.cpp
    types/test_categorical_kind_type.cpp
    types/test_categorical_type.cpp
    types/test_complex_type.cpp
    types/test_convert_type.cpp
    types/test_char_type.cpp
    types/test_cuda_host_type.cpp
    types/test_cuda_device_type.cpp
    types/test_datashape_formatter.cpp
    types/test_datashape_parser.cpp
    types/test_date_type.cpp
    types/test_datetime_type.cpp
    types/test_fixed_dim_type.cpp
    types/test_fixed_bytes_kind_type.cpp
    types/test_fixed_bytes_type.cpp
    types/test_fixed_string_kind_type.cpp
    types/test_fixed_string_type.cpp
    types/test_groupby_type.cpp
    types/test_json_type.cpp
    types/test_option_type.cpp
    types/test_pointer_type.cpp
    types/test_scalar_kind_type.cpp
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
#    types/test_type_promotion.cpp
    types/test_var_dim_type.cpp
    func/special_vals.hpp
    func/test_apply.cpp
    func/test_arithmetic.cpp
    func/test_callable.cpp
    func/test_comparison.cpp
    func/test_compose.cpp
    func/test_compound.cpp
    func/test_constant.cpp
    func/test_elwise.cpp
    func/test_fft.cpp
    func/test_math.cpp
    func/test_max.cpp
    func/test_mean.cpp
    func/test_multidispatch.cpp
    func/test_neighborhood.cpp
    func/test_option.cpp
    func/test_outer.cpp
    func/test_permute.cpp
    func/test_random.cpp
    func/test_reduction.cpp
    func/test_registry.cpp
    func/test_rolling.cpp
    func/test_special.cpp
    func/test_sum.cpp
    func/test_take.cpp
    func/test_take_by_pointer.cpp
    func/test_view.cpp
    gfunc/test_callable.cpp
    array/test_array.cpp
    array/test_array_range.cpp
    array/test_array_assign.cpp
    array/test_array_at.cpp
    array/test_array_cast.cpp
    array/test_array_compare.cpp
    array/test_array_views.cpp
    array/test_asarray.cpp
    array/test_arrmeta_holder.cpp
    array/test_json_formatter.cpp
    array/test_json_parser.cpp
    array/test_memmap.cpp
    array/test_view.cpp
    array/test_with.cpp
    test_bool1.cpp
    test_config.cpp
    test_float16.cpp
    test_integer_sequence.cpp
    test_iterator.cpp
    test_shape_tools.cpp
    test_type_sequence.cpp
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
source_group("Types Tests" REGULAR_EXPRESSION "\\/types\\/test_[^\\/]*\\.cpp$")
source_group("Func Tests" REGULAR_EXPRESSION "\\/func\\/test_[^\\/]*\\.cpp$")

if (DYND_CUDA)
    # Specify that some source files are CUDA files
    foreach(filename_CPP
            func/test_apply.cpp
            func/test_elwise.cpp
            func/test_random.cpp
            func/test_multidispatch.cpp
            types/test_complex_type.cpp
        )
        set_property(SOURCE ${filename_CPP} PROPERTY CUDA_SOURCE_FILE TRUE)
    endforeach(filename_CPP)
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
    if(DYND_SHARED_LIB)
        # Copy libdynd.dll to the directory containing test_libdynd so the tests
        # can find it when they are run.
        add_custom_command(TARGET test_libdynd POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_directory
                           $<TARGET_FILE_DIR:libdynd> $<TARGET_FILE_DIR:test_libdynd>)
    endif()
elseif(APPLE)
    target_link_libraries(test_libdynd
        ${LINK_LIBS}
        )
#    set_target_properties(test_libdynd
 #       PROPERTIES
  #      BUILD_WITH_INSTALL_RPATH TRUE
   #     )
else()
if(LLVM_FOUND)
    set_target_properties(test_libdynd PROPERTIES
    #-Wno-unnamed-type-template-args
        COMPILE_FLAGS "-pthread -Xclang -load -Xclang ${CMAKE_BINARY_DIR}/plugin/libdynd_plugin.so")
else()
    set_target_properties(test_libdynd PROPERTIES
    #-Wno-unnamed-type-template-args
        COMPILE_FLAGS "-pthread")
endif()

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
