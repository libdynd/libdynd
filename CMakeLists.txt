#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

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
    # DTypes
    src/dnd/dtypes/array_dtype.cpp
    src/dnd/dtypes/byteswap_dtype.cpp
    src/dnd/dtypes/conversion_dtype.cpp
    src/dnd/dtypes/dtype_alignment.cpp
    src/dnd/dtypes/fixedstring_dtype.cpp
    src/dnd/dtypes/view_dtype.cpp
    include/dnd/dtypes/array_dtype.hpp
    include/dnd/dtypes/byteswap_dtype.hpp
    include/dnd/dtypes/conversion_dtype.hpp
    include/dnd/dtypes/dtype_alignment.hpp
    include/dnd/dtypes/fixedstring_dtype.hpp
    include/dnd/dtypes/view_dtype.hpp
    # Nodes
    src/dnd/nodes/elementwise_unary_kernel_node.cpp
    src/dnd/nodes/immutable_scalar_node.cpp
    src/dnd/nodes/ndarray_expr_node.cpp
    src/dnd/nodes/ndarray_expr_node_instances.cpp
    src/dnd/nodes/ndarray_expr_node_instances.hpp
    include/dnd/nodes/elementwise_unary_kernel_node.hpp
    include/dnd/nodes/immutable_scalar_node.hpp
    include/dnd/nodes/ndarray_expr_node.hpp
    # Kernels
    src/dnd/kernels/assignment_kernels.cpp
    src/dnd/kernels/builtin_dtype_binary_kernel_table.cpp
    src/dnd/kernels/buffered_unary_kernels.cpp
    src/dnd/kernels/buffered_binary_kernels.cpp
    src/dnd/kernels/byteswap_kernels.cpp
    src/dnd/kernels/string_encoding_kernels.cpp
    src/dnd/kernels/single_assigner_builtin.hpp
    include/dnd/kernels/assignment_kernels.hpp
    include/dnd/kernels/builtin_dtype_binary_kernel_table.hpp
    include/dnd/kernels/buffered_unary_kernels.hpp
    include/dnd/kernels/buffered_binary_kernels.hpp
    include/dnd/kernels/byteswap_kernels.hpp
    include/dnd/kernels/kernel_instance.hpp
    include/dnd/kernels/string_encoding_kernels.hpp
    include/dnd/kernels/unary_kernel_instance.hpp
    # Main
    src/dnd/arithmetic_op.cpp
    src/dnd/dtype.cpp
    src/dnd/dtype_assign.cpp
    src/dnd/dtype_comparisons.cpp
    src/dnd/dtype_promotion.cpp
    src/dnd/exceptions.cpp
    src/dnd/memory_block.cpp
    src/dnd/ndarray.cpp
    src/dnd/ndarray_arange.cpp
    src/dnd/shape_tools.cpp
    include/dnd/auxiliary_data.hpp
    include/dnd/buffer_storage.hpp
    include/dnd/config.hpp
    include/dnd/cling_all.hpp
    include/dnd/diagnostics.hpp
    include/dnd/dtype.hpp
    include/dnd/dtype_comparisons.hpp
    include/dnd/dtype_assign.hpp
    include/dnd/dtype_promotion.hpp
    include/dnd/exceptions.hpp
    include/dnd/fpstatus.hpp
    include/dnd/irange.hpp
    include/dnd/memory_block.hpp
    include/dnd/ndarray.hpp
    include/dnd/ndarray_arange.hpp
    include/dnd/raw_iteration.hpp
    include/dnd/scalars.hpp
    include/dnd/shortvector.hpp
    include/dnd/shape_tools.hpp
    include/dnd/string_encodings.hpp
    )

if(WIN32)
# This is my hardcoded boost path
include_directories(
    D:/Develop/boost_1_48_0/
    )
endif(WIN32)

include_directories(
    include
    thirdparty/utf8/source
    )

source_group("DTypes Source" REGULAR_EXPRESSION "src/dnd/dtypes/.*cpp")
source_group("DTypes Headers" REGULAR_EXPRESSION "include/dnd/dtypes/.*hpp")
source_group("Nodes Source" REGULAR_EXPRESSION "src/dnd/nodes/.*cpp")
source_group("Nodes Headers" REGULAR_EXPRESSION "include/dnd/nodes/.*hpp")
source_group("Kernels Source" REGULAR_EXPRESSION "src/dnd/kernels/.*cpp")
source_group("Kernels Headers" REGULAR_EXPRESSION "include/dnd/kernels/.*hpp")
source_group("Internal Headers" REGULAR_EXPRESSION "src/dnd/.*hpp")

if(WIN32)
    add_library(dynamicndarray STATIC ${dnd_SRC})
else()
    add_library(dynamicndarray SHARED ${dnd_SRC})
endif()

add_subdirectory(tests)

add_subdirectory(examples)

add_subdirectory(python)
