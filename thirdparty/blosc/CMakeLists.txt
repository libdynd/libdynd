cmake_minimum_required(VERSION 2.6)
project(blosc)

set(CMAKE_VERBOSE_MAKEFILE 1)

set(blosc_SRC
    blosc/blosc.c
    blosc/blosclz.c
    blosc/shuffle.c
    blosc/blosc.h
    blosc/blosclz.h
    blosc/shuffle.h
    )

if(WIN32)
    set(blosc_SRC
        ${blosc_SRC}
        blosc/win32/pthread.c
        blosc/win32/pthread.h
        blosc/win32/stdint-windows.h
        )
    include_directories(blosc/win32)
endif()

# Just supporting static library build for now (no .so/.dll yet)
add_library(blosc STATIC ${blosc_SRC})

add_subdirectory(bench)
add_subdirectory(tests)

