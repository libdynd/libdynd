#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(benchmark_libdynd)

add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/benchmark ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/benchmark)

set(benchmarks_SRC
    benchmark_libdynd.cpp
    array/benchmark_empty.cpp
    func/benchmark_apply.cpp
    func/benchmark_arithmetic.cpp
    func/benchmark_random.cpp
    )

include_directories(
    ${CMAKE_SOURCE_DIR}/thirdparty/benchmark/include
    .
    )

if (DYND_CUDA)
    cuda_add_executable(benchmark_libdynd ${benchmarks_SRC})
else()
    add_executable(benchmark_libdynd ${benchmarks_SRC})
endif()

if(WIN32)
    target_link_libraries(benchmark_libdynd
        libdynd
        benchmark
        )
elseif(APPLE)
    target_link_libraries(benchmark_libdynd
        libdynd
        benchmark
        )
else()
    set_target_properties(benchmark_libdynd PROPERTIES
        COMPILE_FLAGS "-pthread")

    target_link_libraries(benchmark_libdynd
        libdynd
        pthread
        benchmark
        )
endif()

# If installation is requested, install the program
if (DYND_INSTALL_LIB)
    install(TARGETS benchmark_libdynd
        RUNTIME DESTINATION bin)
endif()
