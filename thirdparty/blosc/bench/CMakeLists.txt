cmake_minimum_required(VERSION 2.6)
project(blosc_bench)

set(CMAKE_VERBOSE_MAKEFILE 1)

set(bench_SRC
    bench.c
    )

add_executable(blosc_bench ${bench_SRC})

target_link_libraries(blosc_bench
    blosc
    )
