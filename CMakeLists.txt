#
# Copyright (C) 2011-13, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(libdynd)

# Only add these options if this is the top level CMakeLists.txt
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
################################################
# Some options configurable from the CMAKE command execution
#
# -DDYND_SHARED_LIB=ON/OFF, whether to build a shared or a static library.
    if(WIN32)
        option(DYND_SHARED_LIB
            "Build a libdynd shared library instead of a static library"
            OFF)
    else()
        option(DYND_SHARED_LIB
            "Build a libdynd shared library instead of a static library"
            ON)
    endif()
#
# -DDYND_INSTALL_LIB=ON/OFF, whether to install libdynd into the
#   CMAKE_INSTALL_PREFIX. Its main purpose is to allow separate
#   building/linking of libdynd by the dynd-python or other projects.
#   For the make target, use "make install" to do the actual installation.
#   If this is enabled, specify -DUSE_SEPARATE_LIBDYND to the dynd-python
#   cmake command indicating to link against a separately built libdynd
#   instead of trying #   to use one inside the same source tree.
    option(DYND_INSTALL_LIB
        "Do installation of the built libdynd library to the CMAKE_INSTALL_PREFIX"
        OFF)
# -DUSE_RELATIVE_RPATH=ON/OFF, For OSX, to use the @rpath mechanism
#   for creating a build will be linked against with a relative path.
#   Build dynd-python with -DUSE_RELATIVE_PATH=ON as well, and look in
#   the CMakeLists.txt for dynd-python to see how to use this in other
#   projects.
    if(APPLE)
        option(USE_RELATIVE_RPATH
            "OSX: Set the install name of libdynd to '@rpath'."
            OFF)
    endif()
# -DDYND_BUILD_TESTS=ON/OFF, whether to build the googletest unit tests.
    option(DYND_BUILD_TESTS
        "Build the googletest unit tests for libdynd."
        ON)
#
################################################
endif()

list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_VERBOSE_MAKEFILE 1)

# Embedded libraries
add_subdirectory(thirdparty/blosc)
include_directories(thirdparty/blosc/blosc)
add_subdirectory(thirdparty/datetime)
include_directories(thirdparty/datetime/include)

if(WIN32)
    # Treat warnings as errors (-WX does this)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -WX -EHsc")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fomit-frame-pointer -fstrict-aliasing -fPIC -Wall -Wextra -Werror")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # The '-fmax-errors' flag was first supported in gcc 4.6
        # Use the c++0x flag only in 4.6 and later
        if (NOT "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "4.6")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fmax-errors=20")
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=20")
        # Apple has done this weird thing where its Clang reports a different
        # version than the official Clang.
        # TODO: someone more OS X-savvy will have to fine-tune this.
        if (APPLE OR "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "3.2")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wdocumentation")
        endif()
    endif()
endif()

# Get the git revision
include(GetGitRevisionDescriptionDyND)
get_git_head_revision("${CMAKE_CURRENT_SOURCE_DIR}" GIT_REFSPEC DYND_GIT_SHA1)
git_describe("${CMAKE_CURRENT_SOURCE_DIR}" DYND_VERSION_STRING --dirty --match "v[0-9]*")
message(STATUS "DyND version: ${DYND_VERSION_STRING}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dynd/git_version.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/dynd/git_version.cpp" @ONLY)

# Extract the version number from the version string
string(REPLACE "v" "" DYND_VERSION "${DYND_VERSION_STRING}")
string(REPLACE "-" ";" DYND_VERSION "${DYND_VERSION}")
list(GET DYND_VERSION 0 "${DYND_VERSION}")

set(libdynd_SRC
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
    src/dynd/dtypes/base_bytes_dtype.cpp
    src/dynd/dtypes/base_dtype.cpp
    src/dynd/dtypes/base_expression_dtype.cpp
    src/dynd/dtypes/base_string_dtype.cpp
    src/dynd/dtypes/base_struct_dtype.cpp
    src/dynd/dtypes/base_uniform_dim_dtype.cpp
    src/dynd/dtypes/busdate_dtype.cpp
    src/dynd/dtypes/builtin_dtype_properties.cpp
    src/dynd/dtypes/bytes_dtype.cpp
    src/dynd/dtypes/byteswap_dtype.cpp
    src/dynd/dtypes/categorical_dtype.cpp
    src/dynd/dtypes/convert_dtype.cpp
    src/dynd/dtypes/datashape_formatter.cpp
    src/dynd/dtypes/datashape_parser.cpp
    src/dynd/dtypes/date_dtype.cpp
    src/dynd/dtypes/dtype_dtype.cpp
    src/dynd/dtypes/expr_dtype.cpp
    src/dynd/dtypes/property_dtype.cpp
    src/dynd/dtypes/dtype_alignment.cpp
    src/dynd/dtypes/fixed_dim_dtype.cpp
    src/dynd/dtypes/fixedbytes_dtype.cpp
    src/dynd/dtypes/fixedstring_dtype.cpp
    src/dynd/dtypes/fixedstruct_dtype.cpp
    src/dynd/dtypes/groupby_dtype.cpp
    src/dynd/dtypes/json_dtype.cpp
    src/dynd/dtypes/pointer_dtype.cpp
    src/dynd/dtypes/strided_dim_dtype.cpp
    src/dynd/dtypes/string_dtype.cpp
    src/dynd/dtypes/struct_dtype.cpp
    src/dynd/dtypes/tuple_dtype.cpp
    src/dynd/dtypes/type_id.cpp
    src/dynd/dtypes/unary_expr_dtype.cpp
    src/dynd/dtypes/var_dim_dtype.cpp
    src/dynd/dtypes/view_dtype.cpp
    src/dynd/dtypes/void_pointer_dtype.cpp
    include/dynd/dtypes/base_bytes_dtype.hpp
    include/dynd/dtypes/base_dtype.hpp
    include/dynd/dtypes/base_expression_dtype.hpp
    include/dynd/dtypes/base_string_dtype.hpp
    include/dynd/dtypes/base_struct_dtype.hpp
    include/dynd/dtypes/base_uniform_dim_dtype.hpp
    include/dynd/dtypes/busdate_dtype.hpp
    include/dynd/dtypes/builtin_dtype_properties.hpp
    include/dynd/dtypes/bytes_dtype.hpp
    include/dynd/dtypes/byteswap_dtype.hpp
    include/dynd/dtypes/categorical_dtype.hpp
    include/dynd/dtypes/convert_dtype.hpp
    include/dynd/dtypes/datashape_formatter.hpp
    include/dynd/dtypes/datashape_parser.hpp
    include/dynd/dtypes/date_dtype.hpp
    include/dynd/dtypes/dtype_dtype.hpp
    include/dynd/dtypes/expr_dtype.hpp
    include/dynd/dtypes/property_dtype.hpp
    include/dynd/dtypes/dtype_alignment.hpp
    include/dynd/dtypes/fixed_dim_dtype.hpp
    include/dynd/dtypes/fixedbytes_dtype.hpp
    include/dynd/dtypes/fixedstring_dtype.hpp
    include/dynd/dtypes/fixedstruct_dtype.hpp
    include/dynd/dtypes/groupby_dtype.hpp
    include/dynd/dtypes/json_dtype.hpp
    include/dynd/dtypes/pointer_dtype.hpp
    include/dynd/dtypes/strided_dim_dtype.hpp
    include/dynd/dtypes/string_dtype.hpp
    include/dynd/dtypes/struct_dtype.hpp
    include/dynd/dtypes/fixedstruct_dtype.hpp
    include/dynd/dtypes/tuple_dtype.hpp
    include/dynd/dtypes/type_id.hpp
    include/dynd/dtypes/unary_expr_dtype.hpp
    include/dynd/dtypes/var_dim_dtype.hpp
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
    include/dynd/gfunc/call_callable.hpp
    include/dynd/gfunc/elwise_gfunc.hpp
    include/dynd/gfunc/elwise_reduce_gfunc.hpp
    include/dynd/gfunc/make_callable.hpp
    include/dynd/gfunc/serialize.hpp
    # Kernels
    src/dynd/kernels/assignment_kernels.cpp
    src/dynd/kernels/var_dim_assignment_kernels.cpp
    src/dynd/kernels/buffered_binary_kernels.cpp
    src/dynd/kernels/bytes_assignment_kernels.cpp
    src/dynd/kernels/byteswap_kernels.cpp
    src/dynd/kernels/comparison_kernels.cpp
    src/dynd/kernels/date_assignment_kernels.cpp
    src/dynd/kernels/date_expr_kernels.cpp
    src/dynd/kernels/elwise_expr_kernels.cpp
    src/dynd/kernels/expr_kernel_generator.cpp
    src/dynd/kernels/expr_kernels.cpp
    src/dynd/kernels/expression_assignment_kernels.cpp
    src/dynd/kernels/expression_comparison_kernels.cpp
    src/dynd/kernels/string_assignment_kernels.cpp
    src/dynd/kernels/string_algorithm_kernels.cpp
    src/dynd/kernels/string_numeric_assignment_kernels.cpp
    src/dynd/kernels/string_comparison_kernels.cpp
    src/dynd/kernels/struct_assignment_kernels.cpp
    src/dynd/kernels/struct_comparison_kernels.cpp
    src/dynd/kernels/kernels_for_disassembly.cpp
    src/dynd/kernels/single_assigner_builtin.hpp
    src/dynd/kernels/single_comparer_builtin.hpp
    include/dynd/kernels/assignment_kernels.hpp
    include/dynd/kernels/var_dim_assignment_kernels.hpp
    include/dynd/kernels/buffered_binary_kernels.hpp
    include/dynd/kernels/bytes_assignment_kernels.hpp
    include/dynd/kernels/byteswap_kernels.hpp
    include/dynd/kernels/comparison_kernels.hpp
    include/dynd/kernels/date_assignment_kernels.hpp
    include/dynd/kernels/date_expr_kernels.hpp
    include/dynd/kernels/elwise_expr_kernels.hpp
    include/dynd/kernels/expr_kernels.hpp
    include/dynd/kernels/expr_kernel_generator.hpp
    include/dynd/kernels/expression_assignment_kernels.hpp
    include/dynd/kernels/expression_comparison_kernels.hpp
    include/dynd/kernels/hierarchical_kernels.hpp
    include/dynd/kernels/kernel_instance.hpp
    include/dynd/kernels/string_assignment_kernels.hpp
    include/dynd/kernels/string_algorithm_kernels.hpp
    include/dynd/kernels/string_numeric_assignment_kernels.hpp
    include/dynd/kernels/string_comparison_kernels.hpp
    include/dynd/kernels/struct_assignment_kernels.hpp
    include/dynd/kernels/struct_comparison_kernels.hpp
    # MemBlock
    src/dynd/memblock/memory_block.cpp
    src/dynd/memblock/executable_memory_block_windows_x64.cpp
    src/dynd/memblock/executable_memory_block_darwin_x64.cpp
    src/dynd/memblock/executable_memory_block_linux_x64.cpp
    src/dynd/memblock/external_memory_block.cpp
    src/dynd/memblock/fixed_size_pod_memory_block.cpp
    src/dynd/memblock/pod_memory_block.cpp
    src/dynd/memblock/ndobject_memory_block.cpp
    src/dynd/memblock/objectarray_memory_block.cpp
    src/dynd/memblock/zeroinit_memory_block.cpp
    include/dynd/memblock/memory_block.hpp
    include/dynd/memblock/executable_memory_block.hpp
    include/dynd/memblock/external_memory_block.hpp
    include/dynd/memblock/fixed_size_pod_memory_block.hpp
    include/dynd/memblock/pod_memory_block.hpp
    include/dynd/memblock/ndobject_memory_block.hpp
    include/dynd/memblock/objectarray_memory_block.hpp
    include/dynd/memblock/zeroinit_memory_block.hpp
    # VM
    src/dynd/vm/elwise_program.cpp
    src/dynd/vm/register_allocation.cpp
    include/dynd/vm/elwise_program.hpp
    include/dynd/vm/register_allocation.hpp
    # Main
    src/dynd/arithmetic_op.cpp
    src/dynd/dtype.cpp
    src/dynd/dtype_assign.cpp
    src/dynd/dtype_promotion.cpp
    src/dynd/exceptions.cpp
    src/dynd/git_version.cpp.in # Included here for ease of editing in IDEs
    ${CMAKE_CURRENT_BINARY_DIR}/src/dynd/git_version.cpp
    src/dynd/json_formatter.cpp
    src/dynd/json_parser.cpp
    src/dynd/lowlevel_api.cpp
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
    include/dynd/dtype_assign.hpp
    include/dynd/dtype_promotion.hpp
    include/dynd/exceptions.hpp
    include/dynd/fpstatus.hpp
    include/dynd/json_formatter.hpp
    include/dynd/json_parser.hpp
    include/dynd/irange.hpp
    include/dynd/lowlevel_api.hpp
    include/dynd/ndobject.hpp
    include/dynd/ndobject_arange.hpp
    include/dynd/ndobject_iter.hpp
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

if ((NOT DYND_SHARED_LIB) AND DYND_INSTALL_LIB)
    # If we're making an installable static library,
    # include the sublibraries source directly because
    # including static libraries in other static libraries
    # is a hassle
    get_directory_property(datetime_SRC DIRECTORY "thirdparty/datetime" DEFINITION datetime_SRC)
    get_directory_property(blosc_SRC DIRECTORY "thirdparty/blosc" DEFINITION blosc_SRC)
    set(libdynd_SRC
        ${libdynd_SRC}
        ${datetime_SRC}
        ${blosc_SRC}
        )
endif()

if (DYND_SHARED_LIB)
    add_library(libdynd SHARED ${libdynd_SRC})
    set_target_properties(libdynd
        PROPERTIES
        # For now, during development, embed the full version string
        # to reduce the chance of accidentally linking the wrong one.
        OUTPUT_NAME "dynd_${DYND_VERSION_STRING}"
        PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}"
        )
else()
    add_library(libdynd STATIC ${libdynd_SRC})
    set_target_properties(libdynd
        PROPERTIES
        OUTPUT_NAME "dynd"
        PREFIX "${CMAKE_STATIC_LIBRARY_PREFIX}"
        )
endif()

if(WIN32)
    set_target_properties(libdynd
        PROPERTIES
        PREFIX "lib"
        )
elseif(APPLE)
    if(USE_RELATIVE_RPATH)
        message(STATUS "Setting the libdynd install name to @rpath")
        # MW: The INSTALL_NAME_DIR and rpath stuff is a bit confusing,
        #     hopefully someone who really knows this stuff can fix it
        #     up one day.
        set_target_properties(libdynd
            PROPERTIES
            BUILD_WITH_INSTALL_RPATH ON
            INSTALL_NAME_DIR "@rpath"
            )
    endif()
endif()

if (DYND_SHARED_LIB OR (NOT DYND_INSTALL_LIB))
    # If we're not making an installable static library,
    # link the sublibraries normally
    target_link_libraries(libdynd
        datetime
        blosc
        )
endif()

# add_subdirectory(basic_kernels)
if(DYND_BUILD_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(examples)

# Create a libdynd-config script
get_property(dynd_library_prefix TARGET libdynd PROPERTY PREFIX)
get_property(dynd_output_name TARGET libdynd PROPERTY OUTPUT_NAME)
set(DYND_LINK_FLAG "-l${dynd_output_name}")
if (DYND_SHARED_LIB)
    if ("${CMAKE_IMPORT_LIBRARY_SUFFIX}" STREQUAL "")
        set(DYND_LIB_FILE "${dynd_library_prefix}${dynd_output_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    else()
        set(DYND_LIB_FILE "${dynd_library_prefix}${dynd_output_name}${CMAKE_IMPORT_LIBRARY_SUFFIX}")
    endif()
else()
    set(DYND_LIB_FILE "${dynd_library_prefix}${dynd_output_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()
if(WIN32)
    if (DYND_SHARED_LIB)
        set(DYND_STATIC_LIB_DIR "")
    else()
        set(DYND_STATIC_LIB_DIR "\\static")
    endif()
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/libdynd-config.bat.in"
        "${CMAKE_CURRENT_BINARY_DIR}/libdynd-config.bat" @ONLY)
else()
    if (DYND_SHARED_LIB)
        set(DYND_STATIC_LIB_DIR "")
    else()
        set(DYND_STATIC_LIB_DIR "/static")
    endif()
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/libdynd-config.in"
        "${CMAKE_CURRENT_BINARY_DIR}/libdynd-config" @ONLY)
endif()

# If requested, install the .lib/.so/.dll file and headers to the install prefix
if (DYND_INSTALL_LIB)
    # Install the libdynd binary
    install(TARGETS libdynd
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
    # Install the libdynd headers
    install(DIRECTORY "include/dynd" DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
    # Install the libdynd-config script
    if(WIN32)
        install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/libdynd-config.bat"
            DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
    else()
        install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/libdynd-config"
            DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
    endif()
    if(DYND_BUILD_TESTS)
        install(TARGETS test_libdynd
            RUNTIME DESTINATION bin)
    endif()
endif()

