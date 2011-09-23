cmake_minimum_required(VERSION 2.8)
project(dynamicndarray)

set(CMAKE_VERBOSE_MAKEFILE 1)

set(CMAKE_CXX_FLAGS "-std=c++0x -g")
set(LINK_FLAGS -Wall -Wextra -Werror)

set(dnd_SRC
    src/dnd/dtype.cpp
    src/dnd/dtype_assign.cpp
    src/dnd/dtype_casting.cpp
    src/dnd/dtype_promotion.cpp
    src/dnd/exceptions.cpp
    src/dnd/membuffer.cpp
    src/dnd/ndarray.cpp
    src/dnd/shape_tools.cpp
    include/dnd/dtype.hpp
    include/dnd/dtype_assign.hpp
    include/dnd/dtype_casting.hpp
    include/dnd/dtype_promotion.hpp
    include/dnd/exceptions.hpp
    include/dnd/irange.hpp
    include/dnd/membuffer.hpp
    include/dnd/ndarray.hpp
    include/dnd/shape_tools.hpp
    )

include_directories(
    include
    )

add_library(dynamicndarray ${dnd_SRC})

add_subdirectory(tests)
