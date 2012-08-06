#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
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
else()
    set(CMAKE_CXX_FLAGS "-std=c++0x -g -O0 -fomit-frame-pointer -fstrict-aliasing -fPIC -Wall -Wextra -Werror -Weffc++")
    if (CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=20")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=20")
    endif()
endif()
#set(CMAKE_CXX_FLAGS "-std=c++0x -g -O3 -Wall -Wextra -Werror -Weffc++ -fmax-errors=20")
#set(CMAKE_CXX_FLAGS "-std=c++0x -g -fprofile-arcs -ftest-coverage -Wall -Wextra -Werror")
#set(LINK_FLAGS)

set(dnd_SRC
    # CodeGen
    src/dnd/codegen/unary_kernel_adapter_codegen_windows_x64.cpp
    src/dnd/codegen/unary_kernel_adapter_codegen_x64_sysvabi.cpp
    src/dnd/codegen/unary_kernel_adapter_codegen_unsupported.cpp
    src/dnd/codegen/binary_kernel_adapter_codegen_windows_x64.cpp
    src/dnd/codegen/binary_kernel_adapter_codegen_x64_sysvabi.cpp
    src/dnd/codegen/binary_kernel_adapter_codegen_unsupported.cpp
    src/dnd/codegen/binary_reduce_kernel_adapter_codegen.cpp
    src/dnd/codegen/codegen_cache.cpp
    include/dnd/codegen/unary_kernel_adapter_codegen.hpp
    include/dnd/codegen/binary_kernel_adapter_codegen.hpp
    include/dnd/codegen/binary_reduce_kernel_adapter_codegen.hpp
    include/dnd/codegen/calling_conventions.hpp
    include/dnd/codegen/codegen_cache.hpp
    # DTypes
    src/dnd/dtypes/array_dtype.cpp
    src/dnd/dtypes/byteswap_dtype.cpp
    src/dnd/dtypes/categorical_dtype.cpp
    src/dnd/dtypes/convert_dtype.cpp
    src/dnd/dtypes/dtype_alignment.cpp
    src/dnd/dtypes/fixedstring_dtype.cpp
    src/dnd/dtypes/pointer_dtype.cpp
    src/dnd/dtypes/string_dtype.cpp
    src/dnd/dtypes/tuple_dtype.cpp
    src/dnd/dtypes/view_dtype.cpp
    src/dnd/dtypes/void_pointer_dtype.cpp
    include/dnd/dtypes/array_dtype.hpp
    include/dnd/dtypes/byteswap_dtype.hpp
    include/dnd/dtypes/categorical_dtype.hpp
    include/dnd/dtypes/convert_dtype.hpp
    include/dnd/dtypes/dtype_alignment.hpp
    include/dnd/dtypes/fixedstring_dtype.hpp
    include/dnd/dtypes/pointer_dtype.hpp
    include/dnd/dtypes/string_dtype.hpp
    include/dnd/dtypes/tuple_dtype.hpp
    include/dnd/dtypes/view_dtype.hpp
    include/dnd/dtypes/void_pointer_dtype.hpp
    # Eval
    src/dnd/eval/eval_context.cpp
    src/dnd/eval/eval_engine.cpp
    src/dnd/eval/elwise_reduce_eval.cpp
    src/dnd/eval/groupby_elwise_reduce_eval.cpp
    src/dnd/eval/unary_elwise_eval.cpp
    include/dnd/eval/eval_context.hpp
    include/dnd/eval/eval_engine.hpp
    include/dnd/eval/elwise_reduce_eval.hpp
    include/dnd/eval/groupby_elwise_reduce_eval.hpp
    include/dnd/eval/unary_elwise_eval.hpp
    # GFunc
    src/dnd/gfunc/elwise_gfunc.cpp
    src/dnd/gfunc/elwise_reduce_gfunc.cpp
    include/dnd/gfunc/elwise_gfunc.hpp
    include/dnd/gfunc/elwise_reduce_gfunc.hpp
    # Kernels
    src/dnd/kernels/assignment_kernels.cpp
    src/dnd/kernels/builtin_dtype_binary_kernel_table.cpp
    src/dnd/kernels/buffered_unary_kernels.cpp
    src/dnd/kernels/buffered_binary_kernels.cpp
    src/dnd/kernels/byteswap_kernels.cpp
    src/dnd/kernels/string_assignment_kernels.cpp
    src/dnd/kernels/kernels_for_disassembly.cpp
    src/dnd/kernels/single_assigner_builtin.hpp
    include/dnd/kernels/assignment_kernels.hpp
    include/dnd/kernels/builtin_dtype_binary_kernel_table.hpp
    include/dnd/kernels/buffered_unary_kernels.hpp
    include/dnd/kernels/buffered_binary_kernels.hpp
    include/dnd/kernels/byteswap_kernels.hpp
    include/dnd/kernels/kernel_instance.hpp
    include/dnd/kernels/string_assignment_kernels.hpp
    include/dnd/kernels/unary_kernel_instance.hpp
    # MemBlock
    src/dnd/memblock/memory_block.cpp
    src/dnd/memblock/executable_memory_block_windows_x64.cpp
    src/dnd/memblock/executable_memory_block_darwin_x64.cpp
    src/dnd/memblock/executable_memory_block_linux_x64.cpp
    src/dnd/memblock/external_memory_block.cpp
    src/dnd/memblock/fixed_size_pod_memory_block.cpp
    src/dnd/memblock/ndarray_node_memory_block.cpp
    src/dnd/memblock/pod_memory_block.cpp
    include/dnd/memblock/memory_block.hpp
    include/dnd/memblock/executable_memory_block.hpp
    include/dnd/memblock/external_memory_block.hpp
    include/dnd/memblock/fixed_size_pod_memory_block.hpp
    include/dnd/memblock/ndarray_node_memory_block.hpp
    include/dnd/memblock/pod_memory_block.hpp
    # Nodes
    src/dnd/nodes/elwise_unary_kernel_node.cpp
    src/dnd/nodes/elwise_binary_kernel_node.cpp
    src/dnd/nodes/elwise_reduce_kernel_node.cpp
    src/dnd/nodes/groupby_node.cpp
    src/dnd/nodes/immutable_scalar_node.cpp
    src/dnd/nodes/ndarray_node.cpp
    src/dnd/nodes/scalar_node.cpp
    src/dnd/nodes/strided_ndarray_node.cpp
    include/dnd/nodes/elwise_unary_kernel_node.hpp
    include/dnd/nodes/elwise_binary_kernel_node.hpp
    include/dnd/nodes/elwise_reduce_kernel_node.hpp
    include/dnd/nodes/groupby_node.hpp
    include/dnd/nodes/immutable_builtin_scalar_node.hpp
    include/dnd/nodes/immutable_scalar_node.hpp
    include/dnd/nodes/ndarray_node.hpp
    include/dnd/nodes/scalar_node.hpp
    include/dnd/nodes/strided_ndarray_node.hpp
    # Main
    src/dnd/arithmetic_op.cpp
    src/dnd/dtype.cpp
    src/dnd/dtype_assign.cpp
    src/dnd/dtype_comparisons.cpp
    src/dnd/dtype_promotion.cpp
    src/dnd/exceptions.cpp
    src/dnd/ndarray.cpp
    src/dnd/ndarray_arange.cpp
    src/dnd/shape_tools.cpp
    src/dnd/string_encodings.cpp
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
    include/dnd/ndarray.hpp
    include/dnd/ndarray_arange.hpp
    include/dnd/raw_iteration.hpp
    include/dnd/scalars.hpp
    include/dnd/shortvector.hpp
    include/dnd/shape_tools.hpp
    include/dnd/string_encodings.hpp
    include/dnd/platform_definitions.h
    )

if(WIN32)
# This is my hardcoded boost path
include_directories(
    D:/Develop/boost_1_48_0/
    )
else(WIN32)
# Find boost. This works as long as BOOT_ROOT is defined in env
#     (there may be other ways to specify it though)
  find_package(Boost REQUIRED)
  message(STATUS "** Boost Include: ${Boost_INCLUDE_DIR}")
  include_directories(${Boost_INCLUDE_DIR})
endif(WIN32)

include_directories(
    include
    thirdparty/utf8/source
    )

source_group("Main Source" REGULAR_EXPRESSION "src/dnd/.*cpp")
source_group("Main Headers" REGULAR_EXPRESSION "include/dnd/.*hpp")
source_group("CodeGen Source" REGULAR_EXPRESSION "src/dnd/codegen/.*cpp")
source_group("CodeGen Headers" REGULAR_EXPRESSION "include/dnd/codegen/.*hpp")
source_group("DTypes Source" REGULAR_EXPRESSION "src/dnd/dtypes/.*cpp")
source_group("DTypes Headers" REGULAR_EXPRESSION "include/dnd/dtypes/.*hpp")
source_group("Eval Source" REGULAR_EXPRESSION "src/dnd/eval/.*cpp")
source_group("Eval Headers" REGULAR_EXPRESSION "include/dnd/eval/.*hpp")
source_group("GFunc Source" REGULAR_EXPRESSION "src/dnd/gfunc/.*cpp")
source_group("GFunc Headers" REGULAR_EXPRESSION "include/dnd/gfunc/.*hpp")
source_group("Kernels Source" REGULAR_EXPRESSION "src/dnd/kernels/.*cpp")
source_group("Kernels Headers" REGULAR_EXPRESSION "include/dnd/kernels/.*hpp")
source_group("MemBlock Source" REGULAR_EXPRESSION "src/dnd/memblock/.*cpp")
source_group("MemBlock Headers" REGULAR_EXPRESSION "include/dnd/memblock/.*hpp")
source_group("Nodes Source" REGULAR_EXPRESSION "src/dnd/nodes/.*cpp")
source_group("Nodes Headers" REGULAR_EXPRESSION "include/dnd/nodes/.*hpp")
source_group("Internal Headers" REGULAR_EXPRESSION "src/dnd/.*hpp")

if(WIN32)
    add_library(dynamicndarray STATIC ${dnd_SRC})
else()
    add_library(dynamicndarray SHARED ${dnd_SRC})
endif()

add_subdirectory(basic_kernels)

add_subdirectory(tests)

add_subdirectory(examples)

# should we use build the python support?
option(
    BUILD_PYTHON
    "Build python bindings" ON
    )

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()
