cmake_minimum_required(VERSION 2.8)
project(dynamicndarray)

set(CMAKE_VERBOSE_MAKEFILE 1)
set(BUILD_SHARED_LIBS 1)

# Clang Compiler
#include(CMakeForceCompiler)
#cmake_force_c_compiler(/usr/local/bin/clang CLang)
#cmake_force_cxx_compiler(/usr/local/bin/clang++ CLang)

if(WIN32)
	# Treat warnings as errors (-WX does this)
	set(CMAKE_CXX_FLAGS "-WX -EHsc")
else(WIN32)
	set(CMAKE_CXX_FLAGS "-std=c++0x -g -O0 -fPIC -Wall -Wextra -Werror -Weffc++ -fmax-errors=20")
endif(WIN32)
#set(CMAKE_CXX_FLAGS "-std=c++0x -g -O3 -Wall -Wextra -Werror -Weffc++ -fmax-errors=20")
#set(CMAKE_CXX_FLAGS "-std=c++0x -g -fprofile-arcs -ftest-coverage -Wall -Wextra -Werror")
set(LINK_FLAGS)

set(dnd_SRC
    src/dnd/array_dtype.cpp
    src/dnd/dtype.cpp
    src/dnd/dtype_assign.cpp
    src/dnd/dtype_casting.cpp
    src/dnd/dtype_promotion.cpp
    src/dnd/exceptions.cpp
    src/dnd/ndarray.cpp
    src/dnd/ndarray_arange.cpp
    src/dnd/ndarray_expr_node.cpp
    src/dnd/ndarray_expr_node_instances.cpp
    src/dnd/arithmetic_op.cpp
    src/dnd/shape_tools.cpp
    src/dnd/ndarray_expr_node_instances.hpp
    src/dnd/single_assigner_simple.hpp
    include/dnd/array_dtype.hpp
    include/dnd/config.hpp
    include/dnd/dtype.hpp
    include/dnd/dtype_assign.hpp
    include/dnd/dtype_casting.hpp
    include/dnd/dtype_promotion.hpp
    include/dnd/exceptions.hpp
    include/dnd/irange.hpp
    include/dnd/ndarray.hpp
    include/dnd/ndarray_arange.hpp
    include/dnd/ndarray_expr_node.hpp
    include/dnd/raw_iteration.hpp
    include/dnd/shape_tools.hpp
    )

if(WIN32)
# This is my hardcoded boost path
include_directories(
    D:/Develop/boost_1_48_0/
    )
endif(WIN32)

include_directories(
    include
    )

if(WIN32)
    add_library(dynamicndarray STATIC ${dnd_SRC})
else(WIN32)
    add_library(dynamicndarray SHARED ${dnd_SRC})
endif(WIN32)

add_subdirectory(tests)

add_subdirectory(examples)
