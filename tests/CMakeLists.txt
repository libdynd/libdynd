cmake_minimum_required(VERSION 2.8)
project(dnd_tests)

set(tests_SRC
    test_dtype_assign.cpp
    test_dtype_casting.cpp
    test_dtype.cpp
    test_ndarray_assign.cpp
    test_ndarray.cpp
    )

# TODO: this is awkwardly hardcoded
include_directories(
    ../../AnimPhys/gtest/include
    ../include
    )

add_executable(dnd_tests ${tests_SRC})

# gtest generates "unnamed-type-template-args" warnings with clang
set_target_properties(dnd_tests
                PROPERTIES COMPILE_FLAGS
                "-Wno-unnamed-type-template-args -pthread")

target_link_libraries(dnd_tests
    dynamicndarray
    pthread
    /home/mwiebe/Develop/AnimPhys/gtest/make/gtest-all.o
    /home/mwiebe/Develop/AnimPhys/gtest/make/gtest_main.o
    )
