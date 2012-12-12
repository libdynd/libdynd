#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(dynamicndarray)

set(CMAKE_VERBOSE_MAKEFILE 1)
set(BUILD_SHARED_LIBS 1)

# Embedded libraries
add_subdirectory(thirdparty/blosc)
include_directories(thirdparty/blosc/blosc)
add_subdirectory(thirdparty/datetime)
include_directories(thirdparty/datetime/include)

# If we're using gcc, check the version
if (CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                    OUTPUT_VARIABLE GCC_VERSION)
endif()

if(WIN32)
    # Treat warnings as errors (-WX does this)
    set(CMAKE_CXX_FLAGS "-WX -EHsc")
else()
    set(CMAKE_CXX_FLAGS "-g -O0 -fomit-frame-pointer -fstrict-aliasing -fPIC -Wall -Wextra -Werror")
    if (CMAKE_COMPILER_IS_GNUCXX)
        # The '-fmax-errors' flag was first supported in gcc 4.6
        # Use the c++0x flag only in 4.6 and later
        if (NOT GCC_VERSION VERSION_LESS 4.6)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fmax-errors=20")
        endif()
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -ferror-limit=20")
    endif()
endif()

set(dynd_SRC
    # CodeGen
    src/dynd/codegen/unary_kernel_adapter_codegen_windows_x64.cpp
    src/dynd/codegen/unary_kernel_adapter_codegen_x64_sysvabi.cpp
    src/dynd/codegen/unary_kernel_adapter_codegen_unsupported.cpp
    src/dynd/codegen/binary_kernel_adapter_codegen_windows_x64.cpp
    src/dynd/codegen/binary_kernel_adapter_codegen_x64_sysvabi.cpp
    src/dynd/codegen/binary_kernel_adapter_codegen_unsupported.cpp
    src/dynd/codegen/binary_reduce_kernel_adapter_codegen.cpp
    src/dynd/codegen/codegen_cache.cpp
    include/dynd/codegen/unary_kernel_adapter_codegen.hpp
    include/dynd/codegen/binary_kernel_adapter_codegen.hpp
    include/dynd/codegen/binary_reduce_kernel_adapter_codegen.hpp
    include/dynd/codegen/calling_conventions.hpp
    include/dynd/codegen/codegen_cache.hpp
    # DTypes
    src/dynd/dtypes/array_dtype.cpp
    src/dynd/dtypes/busdate_dtype.cpp
    src/dynd/dtypes/bytes_dtype.cpp
    src/dynd/dtypes/byteswap_dtype.cpp
    src/dynd/dtypes/categorical_dtype.cpp
    src/dynd/dtypes/convert_dtype.cpp
    src/dynd/dtypes/date_dtype.cpp
    src/dynd/dtypes/date_property_dtype.cpp
    src/dynd/dtypes/dtype_alignment.cpp
    src/dynd/dtypes/fixedarray_dtype.cpp
    src/dynd/dtypes/fixedbytes_dtype.cpp
    src/dynd/dtypes/fixedstring_dtype.cpp
    src/dynd/dtypes/fixedstruct_dtype.cpp
    src/dynd/dtypes/pointer_dtype.cpp
    src/dynd/dtypes/strided_array_dtype.cpp
    src/dynd/dtypes/string_dtype.cpp
    src/dynd/dtypes/struct_dtype.cpp
    src/dynd/dtypes/tuple_dtype.cpp
    src/dynd/dtypes/view_dtype.cpp
    src/dynd/dtypes/void_pointer_dtype.cpp
    include/dynd/dtypes/array_dtype.hpp
    include/dynd/dtypes/busdate_dtype.hpp
    include/dynd/dtypes/bytes_dtype.hpp
    include/dynd/dtypes/byteswap_dtype.hpp
    include/dynd/dtypes/categorical_dtype.hpp
    include/dynd/dtypes/convert_dtype.hpp
    include/dynd/dtypes/date_dtype.hpp
    include/dynd/dtypes/date_property_dtype.hpp
    include/dynd/dtypes/dtype_alignment.hpp
    include/dynd/dtypes/fixedarray_dtype.hpp
    include/dynd/dtypes/fixedbytes_dtype.hpp
    include/dynd/dtypes/fixedstring_dtype.hpp
    include/dynd/dtypes/fixedstruct_dtype.hpp
    include/dynd/dtypes/pointer_dtype.hpp
    include/dynd/dtypes/strided_array_dtype.hpp
    include/dynd/dtypes/string_dtype.hpp
    include/dynd/dtypes/struct_dtype.hpp
    include/dynd/dtypes/fixedstruct_dtype.hpp
    include/dynd/dtypes/tuple_dtype.hpp
    include/dynd/dtypes/view_dtype.hpp
    include/dynd/dtypes/void_pointer_dtype.hpp
    # Eval
    src/dynd/eval/eval_context.cpp
    src/dynd/eval/eval_elwise_vm.cpp
    src/dynd/eval/eval_engine.cpp
    src/dynd/eval/elwise_reduce_eval.cpp
    src/dynd/eval/groupby_elwise_reduce_eval.cpp
    src/dynd/eval/unary_elwise_eval.cpp
    include/dynd/eval/eval_context.hpp
    include/dynd/eval/eval_elwise_vm.hpp
    include/dynd/eval/eval_engine.hpp
    include/dynd/eval/elwise_reduce_eval.hpp
    include/dynd/eval/groupby_elwise_reduce_eval.hpp
    include/dynd/eval/unary_elwise_eval.hpp
    # GFunc
    src/dynd/gfunc/callable.cpp
    src/dynd/gfunc/elwise_gfunc.cpp
    src/dynd/gfunc/elwise_reduce_gfunc.cpp
    src/dynd/gfunc/serialize.cpp
    include/dynd/gfunc/callable.hpp
    include/dynd/gfunc/elwise_gfunc.hpp
    include/dynd/gfunc/elwise_reduce_gfunc.hpp
    include/dynd/gfunc/make_callable.hpp
    include/dynd/gfunc/serialize.hpp
    # Kernels
    src/dynd/kernels/assignment_kernels.cpp
    src/dynd/kernels/array_assignment_kernels.cpp
    src/dynd/kernels/builtin_dtype_binary_kernel_table.cpp
    src/dynd/kernels/buffered_unary_kernels.cpp
    src/dynd/kernels/buffered_binary_kernels.cpp
    src/dynd/kernels/bytes_assignment_kernels.cpp
    src/dynd/kernels/byteswap_kernels.cpp
    src/dynd/kernels/date_assignment_kernels.cpp
    src/dynd/kernels/string_assignment_kernels.cpp
    src/dynd/kernels/string_numeric_assignment_kernels.cpp
    src/dynd/kernels/struct_assignment_kernels.cpp
    src/dynd/kernels/kernels_for_disassembly.cpp
    src/dynd/kernels/single_assigner_builtin.hpp
    include/dynd/kernels/assignment_kernels.hpp
    include/dynd/kernels/array_assignment_kernels.hpp
    include/dynd/kernels/builtin_dtype_binary_kernel_table.hpp
    include/dynd/kernels/buffered_unary_kernels.hpp
    include/dynd/kernels/buffered_binary_kernels.hpp
    include/dynd/kernels/bytes_assignment_kernels.hpp
    include/dynd/kernels/byteswap_kernels.hpp
    include/dynd/kernels/date_assignment_kernels.hpp
    include/dynd/kernels/kernel_instance.hpp
    include/dynd/kernels/single_compare_kernel_instance.hpp
    include/dynd/kernels/string_assignment_kernels.hpp
    include/dynd/kernels/string_numeric_assignment_kernels.hpp
    include/dynd/kernels/struct_assignment_kernels.hpp
    # MemBlock
    src/dynd/memblock/memory_block.cpp
    src/dynd/memblock/executable_memory_block_windows_x64.cpp
    src/dynd/memblock/executable_memory_block_darwin_x64.cpp
    src/dynd/memblock/executable_memory_block_linux_x64.cpp
    src/dynd/memblock/external_memory_block.cpp
    src/dynd/memblock/fixed_size_pod_memory_block.cpp
    src/dynd/memblock/pod_memory_block.cpp
    src/dynd/memblock/ndobject_memory_block.cpp
    include/dynd/memblock/memory_block.hpp
    include/dynd/memblock/executable_memory_block.hpp
    include/dynd/memblock/external_memory_block.hpp
    include/dynd/memblock/fixed_size_pod_memory_block.hpp
    include/dynd/memblock/pod_memory_block.hpp
    include/dynd/memblock/ndobject_memory_block.hpp
    # VM
    src/dynd/vm/elwise_program.cpp
    src/dynd/vm/register_allocation.cpp
    include/dynd/vm/elwise_program.hpp
    include/dynd/vm/register_allocation.hpp
    # Main
    src/dynd/arithmetic_op.cpp
    src/dynd/dtype.cpp
    src/dynd/dtype_assign.cpp
    src/dynd/dtype_comparisons.cpp
    src/dynd/dtype_promotion.cpp
    src/dynd/exceptions.cpp
    src/dynd/ndobject.cpp
    src/dynd/ndobject_arange.cpp
    src/dynd/shape_tools.cpp
    src/dynd/string_encodings.cpp
    include/dynd/atomic_refcount.hpp
    include/dynd/auxiliary_data.hpp
    include/dynd/buffer_storage.hpp
    include/dynd/config.hpp
    include/dynd/cling_all.hpp
    include/dynd/diagnostics.hpp
    include/dynd/dtype.hpp
    include/dynd/dtype_comparisons.hpp
    include/dynd/dtype_assign.hpp
    include/dynd/dtype_promotion.hpp
    include/dynd/exceptions.hpp
    include/dynd/fpstatus.hpp
    include/dynd/irange.hpp
    include/dynd/ndobject.hpp
    include/dynd/ndobject_arange.hpp
    include/dynd/ndobject_iter.hpp
    include/dynd/raw_iteration.hpp
    include/dynd/scalars.hpp
    include/dynd/shortvector.hpp
    include/dynd/shape_tools.hpp
    include/dynd/string_encodings.hpp
    include/dynd/platform_definitions.h
    )

include_directories(
    include
    thirdparty/utf8/source
    )

source_group("Main Source" REGULAR_EXPRESSION "src/dynd/.*cpp")
source_group("Main Headers" REGULAR_EXPRESSION "include/dynd/.*hpp")
source_group("CodeGen Source" REGULAR_EXPRESSION "src/dynd/codegen/.*cpp")
source_group("CodeGen Headers" REGULAR_EXPRESSION "include/dynd/codegen/.*hpp")
source_group("DTypes Source" REGULAR_EXPRESSION "src/dynd/dtypes/.*cpp")
source_group("DTypes Headers" REGULAR_EXPRESSION "include/dynd/dtypes/.*hpp")
source_group("Eval Source" REGULAR_EXPRESSION "src/dynd/eval/.*cpp")
source_group("Eval Headers" REGULAR_EXPRESSION "include/dynd/eval/.*hpp")
source_group("GFunc Source" REGULAR_EXPRESSION "src/dynd/gfunc/.*cpp")
source_group("GFunc Headers" REGULAR_EXPRESSION "include/dynd/gfunc/.*hpp")
source_group("Kernels Source" REGULAR_EXPRESSION "src/dynd/kernels/.*cpp")
source_group("Kernels Headers" REGULAR_EXPRESSION "include/dynd/kernels/.*hpp")
source_group("MemBlock Source" REGULAR_EXPRESSION "src/dynd/memblock/.*cpp")
source_group("MemBlock Headers" REGULAR_EXPRESSION "include/dynd/memblock/.*hpp")
source_group("VM Source" REGULAR_EXPRESSION "src/dynd/vm/.*cpp")
source_group("VM Headers" REGULAR_EXPRESSION "include/dynd/vm/.*hpp")
source_group("Internal Headers" REGULAR_EXPRESSION "src/dynd/.*hpp")

if(WIN32)
    add_library(dynamicndarray STATIC ${dynd_SRC})
else()
    add_library(dynamicndarray SHARED ${dynd_SRC})
endif()
target_link_libraries(dynamicndarray
    datetime
    )

add_subdirectory(basic_kernels)

add_subdirectory(tests)

add_subdirectory(examples)

