cmake_minimum_required(VERSION 2.8)
project(dynamicndarray)

set(CMAKE_VERBOSE_MAKEFILE 1)

set(dnd_SRC
    src/dnd/dtype.cpp
    include/dnd/dtype.hpp
    )

include_directories(
    include
    )

add_library(dynamicndarray ${dnd_SRC})

add_subdirectory(tests)
