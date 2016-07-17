#
# Copyright (C) 2011-16 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.8.11)
project(libdynd)

include(CheckTypeSize)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(LIB_SUFFIX "" CACHE STRING
    "Typically an empty string or 64. Controls installation to lib or lib64")

add_definitions(-DDYND_SHARED_LIBRARY_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}")


CHECK_TYPE_SIZE("float" SIZEOF_FLOAT)
if(NOT (SIZEOF_FLOAT EQUAL 4))
  message(FATAL_ERROR "libdynd requires sizeof(float) == 4")
endif()

CHECK_TYPE_SIZE("double" SIZEOF_DOUBLE)
if(NOT (SIZEOF_DOUBLE EQUAL 8))
  message(FATAL_ERROR "libdynd requires sizeof(double) == 8")
endif()

# Only add these options if this is the top level CMakeLists.txt
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
################################################
# Some options configurable from the CMAKE command execution
#
# -DDYND_SHARED_LIB=ON/OFF, whether to build a shared or a static library.
    option(DYND_SHARED_LIB
           "Build a libdynd shared library instead of a static library"
           ON)
# -DDYND_LLVM=ON/OFF, whether to build libdynd with or without LLVM support
    option(DYND_LLVM
           "Build a libdynd library with LLVM"
           OFF)
#
# -DDYND_INSTALL_LIB=ON/OFF, whether to install libdynd into the
#   CMAKE_INSTALL_PREFIX. Its main purpose is to allow dynd-python and
#   libdynd to be built inside the source tree of another project, like
#   in the libraries/libdynd subdirectory of dynd-python.
    option(DYND_INSTALL_LIB
           "Do installation of the built libdynd library to the CMAKE_INSTALL_PREFIX"
           ON)
# -DDYND_BUILD_TESTS=ON/OFF, whether to build the googletest unit tests.
    option(DYND_BUILD_TESTS
           "Build the googletest unit tests for libdynd."
           ON)
# -DDYND_BUILD_BENCHMARKS=ON/OFF, whether to build the Google benchmarks
    option(DYND_BUILD_BENCHMARKS
           "Build the Google benchmarks for libdynd."
           ON)
# -DDYND_BUILD_PLUGINS=ON/OFF, whether to build the DyND plugins
    option(DYND_BUILD_PLUGINS
           "Build the plugins for libdynd."
           OFF)
# -DDYND_BUILD_DOCS=ON/OFF, whether or not to build the documentation.
    option(DYND_BUILD_DOCS
           "Use Doxygen to generate the documentation."
           OFF)
# -DDYND_COVERAGE=ON/OFF, whether to generate test coverage information
    option(DYND_COVERAGE
           "Generate code coverage reports from the unit test suite."
           OFF)
#
################################################
endif()

if(DYND_COVERAGE)
  INCLUDE(CodeCoverage)

  SET(CMAKE_BUILD_TYPE "Debug")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_COVERAGE}")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_COVERAGE}")
endif()

if(DYND_LLVM)
  find_package(LLVM CONFIG)
endif()

list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_VERBOSE_MAKEFILE 1)

if(WIN32)
    set(DYND_BUILD_BENCHMARKS OFF)
endif()

if (DEFINED CMAKE_CXX_SIMULATED_ID)
    set(DYND_SIMULATED_COMPILER ${CMAKE_CXX_SIMULATED_ID})
else()
    set(DYND_SIMULATED_COMPILER ${CMAKE_CXX_COMPILER_ID})
endif()

if("${MSVC}" OR "${DYND_SIMULATED_COMPILER}" STREQUAL "MSVC")
    # -WX: Treat warnings as errors
    # -bigobj: Allow lots of symbols (assignment_kernels.cpp and assignment_kernels.cu need this flag)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # If this is the clang-cl compiler.
        # Don't enable -WX here yet.
        set(CMAKE_CXX_FLAGS "-EHsc -bigobj -wd4503 ${CMAKE_CXX_FLAGS}")
    elseif(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 19)
        message(FATAL_ERROR "Only MSVC 2015 (Version 19.0) and later are supported by LibDyND. Found version ${CMAKE_CXX_COMPILER_VERSION}.")
    else()
        set(CMAKE_CXX_FLAGS "-WX -EHsc -bigobj -wd4503 ${CMAKE_CXX_FLAGS}")
    endif()
else()
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 4.9)
            message(FATAL_ERROR "Only GCC 4.9 and later are supported by LibDyND. Found version ${CMAKE_CXX_COMPILER_VERSION}.")
        endif()
        set(CMAKE_CXX_FLAGS "-std=c++14 -fmax-errors=20 -Wno-type-limits -Wno-maybe-uninitialized ${CMAKE_CXX_FLAGS}")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "-std=c++14 -ferror-limit=20 -Wno-missing-braces -ftemplate-depth=500 ${CMAKE_CXX_FLAGS}")
    endif()

    if(WIN32)
        set(CMAKE_CXX_FLAGS "-O3 -std=c++14 -fomit-frame-pointer -fstrict-aliasing -Wall -Wextra -Werror -Wno-missing-field-initializers ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "-O3 -fomit-frame-pointer -fstrict-aliasing -Wall -Wextra -Wno-missing-field-initializers -fPIC -Werror -Wno-ignored-attributes ${CMAKE_CXX_FLAGS}")
    endif()
endif()

if("${UNIX}" AND NOT CMAKE_SYSTEM_NAME MATCHES "BSD")
    set(DYNDT_LINK_LIBS ${DYNDT_LINK_LIBS} dl)
endif()

# LLVM, disabled for now
#add_definitions(${LLVM_DEFINITIONS})
#include_directories(${LLVM_INCLUDE_DIRS})
#llvm_map_components_to_libnames(LLVM_LINK_LIBS core option target bitreader support profiledata codegen irreader linker instrumentation objcarcopts lto)

# Get the git revision
include(GetGitRevisionDescriptionDyND)
get_git_head_revision("${CMAKE_CURRENT_SOURCE_DIR}" GIT_REFSPEC DYND_GIT_SHA1)
git_describe("${CMAKE_CURRENT_SOURCE_DIR}" DYND_VERSION_STRING
             --dirty --always --match "v*")
message(STATUS "DyND version: ${DYND_VERSION_STRING}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dynd/git_version.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/dynd/git_version.cpp" @ONLY)

if(DYND_SHARED_LIB)
    set(DYND_SHARED_LIB_VAL 1)
else()
    set(DYND_SHARED_LIB_VAL 0)
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/dynd/visibility.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/dynd/visibility.hpp" @ONLY)

# Extract the version number from the version string
string(REPLACE "v" "" DYND_VERSION "${DYND_VERSION_STRING}")
string(REPLACE "-" ";" DYND_VERSION "${DYND_VERSION}")
list(GET DYND_VERSION 0 "${DYND_VERSION}")

set(libdyndt_SRC
    # Eval
    src/dynd/eval/eval_context.cpp
    include/dynd/eval/eval_context.hpp
    # Types
    src/dynd/types/any_kind_type.cpp
    src/dynd/types/array_type.cpp
    src/dynd/types/base_type.cpp
    src/dynd/types/base_dim_type.cpp
    src/dynd/types/base_expr_type.cpp
    src/dynd/types/base_memory_type.cpp
    src/dynd/types/base_string_type.cpp
    src/dynd/types/bytes_type.cpp
    src/dynd/types/categorical_kind_type.cpp
    src/dynd/types/callable_type.cpp
    src/dynd/types/char_type.cpp
    src/dynd/types/complex_kind_type.cpp
    src/dynd/types/cuda_host_type.cpp
    src/dynd/types/cuda_device_type.cpp
    src/dynd/types/datashape_formatter.cpp
    src/dynd/types/datashape_parser.cpp
    src/dynd/types/dim_fragment_type.cpp
    src/dynd/types/ellipsis_dim_type.cpp
    src/dynd/types/fixed_bytes_kind_type.cpp
    src/dynd/types/fixed_bytes_type.cpp
    src/dynd/types/fixed_dim_type.cpp
    src/dynd/types/fixed_dim_kind_type.cpp
    src/dynd/types/fixed_string_kind_type.cpp
    src/dynd/types/fixed_string_type.cpp
    src/dynd/types/float_kind_type.cpp
    src/dynd/types/int_kind_type.cpp
    src/dynd/types/option_type.cpp
    src/dynd/types/pointer_type.cpp
    src/dynd/types/pow_dimsym_type.cpp
    src/dynd/types/scalar_kind_type.cpp
    src/dynd/types/state_type.cpp
    src/dynd/types/string_type.cpp
    src/dynd/types/struct_type.cpp
    src/dynd/types/substitute_typevars.cpp
    src/dynd/types/tuple_type.cpp
    src/dynd/types/type_id.cpp
    src/dynd/types/type_type.cpp
    src/dynd/types/typevar_type.cpp
    src/dynd/types/typevar_constructed_type.cpp
    src/dynd/types/typevar_dim_type.cpp
    src/dynd/types/uint_kind_type.cpp
    src/dynd/types/var_dim_type.cpp
    include/dynd/types/any_kind_type.hpp
    include/dynd/types/array_type.hpp
    include/dynd/types/base_bytes_type.hpp
    include/dynd/types/base_type.hpp
    include/dynd/types/base_dim_type.hpp
    include/dynd/types/base_string_type.hpp
    include/dynd/types/bytes_type.hpp
    include/dynd/types/callable_type.hpp
    include/dynd/types/char_type.hpp
    include/dynd/types/datashape_formatter.hpp
    include/dynd/types/datashape_parser.hpp
    include/dynd/types/fixed_bytes_type.hpp
    include/dynd/types/fixed_dim_type.hpp
    include/dynd/types/fixed_dim_kind_type.hpp
    include/dynd/types/sso_bytestring.hpp
    include/dynd/types/state_type.hpp
    include/dynd/types/string_type.hpp
    include/dynd/types/struct_type.hpp
    include/dynd/types/substitute_typevars.hpp
    include/dynd/types/tuple_type.hpp
    include/dynd/types/type_id.hpp
    include/dynd/types/type_type.hpp
    # Memory blocks
    src/dynd/memblock/base_memory_block.cpp
    include/dynd/memblock/buffer_memory_block.hpp
    include/dynd/memblock/base_memory_block.hpp
    include/dynd/memblock/external_memory_block.hpp
    include/dynd/memblock/fixed_size_pod_memory_block.hpp
    include/dynd/memblock/memmap_memory_block.hpp
    include/dynd/memblock/objectarray_memory_block.hpp
    include/dynd/memblock/pod_memory_block.hpp
    include/dynd/memblock/zeroinit_memory_block.hpp
    # Main
    src/dynd/buffer.cpp
    src/dynd/config.cpp
    src/dynd/exceptions.cpp
    src/dynd/float16.cpp
    src/dynd/float128.cpp
    src/dynd/git_version.cpp.in # Included here for ease of editing in IDEs
    ${CMAKE_CURRENT_BINARY_DIR}/src/dynd/git_version.cpp
    src/dynd/int128.cpp
    src/dynd/parse_util.cpp
    src/dynd/shape_tools.cpp
    src/dynd/string_encodings.cpp
    src/dynd/type.cpp
    src/dynd/type_promotion.cpp
    src/dynd/type_registry.cpp
    src/dynd/uint128.cpp
    include/dynd/bytes.hpp
    include/dynd/float16.hpp
    include/dynd/float128.hpp
    include/dynd/int128.hpp
    include/dynd/exceptions.hpp
    include/dynd/parse.hpp
    include/dynd/shape_tools.hpp
    include/dynd/string_encodings.hpp
    include/dynd/type.hpp
    include/dynd/type_registry.hpp
    include/dynd/uint128.hpp
)

set(libdynd_SRC
    # Types
    src/dynd/types/adapt_type.cpp
    src/dynd/types/categorical_type.cpp
    src/dynd/types/substitute_shape.cpp
    include/dynd/types/adapt_type.hpp
    include/dynd/types/categorical_type.hpp
    include/dynd/types/substitute_shape.hpp
    # Callables
    src/dynd/callables/base_callable.cpp
    include/dynd/callables/base_callable.hpp
    include/dynd/callables/base_dispatch_callable.hpp
    # Kernels
    src/dynd/kernels/byteswap_kernels.cpp
    src/dynd/kernels/kernel_builder.cpp
    include/dynd/kernels/apply.hpp
    include/dynd/kernels/arithmetic.hpp
    include/dynd/kernels/assign_na_kernel.hpp
    include/dynd/kernels/assignment_kernels.hpp
    include/dynd/kernels/base_kernel.hpp
    include/dynd/kernels/byteswap_kernels.hpp
    include/dynd/kernels/compose_kernel.hpp
    include/dynd/kernels/compound_kernel.hpp
    include/dynd/kernels/constant_kernel.hpp
    include/dynd/kernels/cuda_launch.hpp
    include/dynd/kernels/dereference_kernel.hpp
    include/dynd/kernels/elwise_kernel.hpp
    include/dynd/kernels/index_kernel.hpp
    include/dynd/kernels/is_na_kernel.hpp
    include/dynd/kernels/kernel_builder.hpp
    include/dynd/kernels/kernel_prefix.hpp
    include/dynd/kernels/max_kernel.hpp
    include/dynd/kernels/min_kernel.hpp
    include/dynd/kernels/reduction_kernel.hpp
    include/dynd/kernels/serialize_kernel.hpp
    include/dynd/kernels/sort_kernel.hpp
    include/dynd/kernels/string_concat_kernel.hpp
    include/dynd/kernels/string_count_kernel.hpp
    include/dynd/kernels/string_find_kernel.hpp
    include/dynd/kernels/string_rfind_kernel.hpp
    include/dynd/kernels/string_replace_kernel.hpp
    include/dynd/kernels/string_startswith_kernel.hpp
    include/dynd/kernels/string_endswith_kernel.hpp
    include/dynd/kernels/string_contains_kernel.hpp
    include/dynd/kernels/take_kernel.hpp
    include/dynd/kernels/tuple_assignment_kernels.hpp
    include/dynd/kernels/uniform_kernel.hpp
    include/dynd/kernels/view_kernel.hpp
    # Main
    src/dynd/access.cpp
    src/dynd/add.cpp
    src/dynd/array.cpp
    src/dynd/array_range.cpp
    src/dynd/asarray.cpp
    src/dynd/assignment.cpp
    src/dynd/bitwise_and.cpp
    src/dynd/bitwise_not.cpp
    src/dynd/bitwise_or.cpp
    src/dynd/bitwise_xor.cpp
    src/dynd/callable.cpp
    src/dynd/cbrt.cpp
    src/dynd/comparison.cpp
    src/dynd/compound_add.cpp
    src/dynd/compound_div.cpp
    src/dynd/convert.cpp
    src/dynd/divide.cpp
    src/dynd/functional.cpp
    src/dynd/index.cpp
    src/dynd/io.cpp
    src/dynd/json_formatter.cpp
    src/dynd/json_parser.cpp
    src/dynd/left_shift.cpp
    src/dynd/limits.cpp
    src/dynd/logic.cpp
    src/dynd/logical_and.cpp
    src/dynd/logical_not.cpp
    src/dynd/logical_or.cpp
    src/dynd/logical_xor.cpp
    src/dynd/math.cpp
    src/dynd/minus.cpp
    src/dynd/mod.cpp
    src/dynd/multiply.cpp
    src/dynd/option.cpp
    src/dynd/parse.cpp
    src/dynd/plus.cpp
    src/dynd/pointer.cpp
    src/dynd/pow.cpp
    src/dynd/random.cpp
    src/dynd/range.cpp
    src/dynd/registry.cpp
    src/dynd/right_shift.cpp
    src/dynd/search.cpp
    src/dynd/sort.cpp
    src/dynd/sqrt.cpp
    src/dynd/statistics.cpp
    src/dynd/string.cpp
    src/dynd/subtract.cpp
    src/dynd/sum.cpp
    src/dynd/view.cpp
    include/dynd/access.hpp
    include/dynd/arithmetic.hpp
    include/dynd/array.hpp
    include/dynd/array_range.hpp
    include/dynd/array_iter.hpp
    include/dynd/arrmeta_holder.hpp
    include/dynd/asarray.hpp
    include/dynd/assignment.hpp
    include/dynd/callable.hpp
    include/dynd/cmake_config.hpp.in # Included here for ease of editing in IDEs
    ${CMAKE_CURRENT_BINARY_DIR}/include/dynd/cmake_config.hpp
    include/dynd/comparison.hpp
    include/dynd/complex.hpp
    include/dynd/config.hpp
    include/dynd/cling_all.hpp
    include/dynd/convert.hpp
    include/dynd/diagnostics.hpp
    include/dynd/ensure_immutable_contig.hpp
    include/dynd/func/elwise.hpp
    include/dynd/func/reduction.hpp
    include/dynd/functional.hpp
    include/dynd/io.hpp
    include/dynd/iterator.hpp
    include/dynd/logic.hpp
    include/dynd/math.hpp
    include/dynd/random.hpp
    include/dynd/range.hpp
    include/dynd/registry.hpp
    include/dynd/sort.hpp
    include/dynd/statistics.hpp
    include/dynd/string.hpp
    include/dynd/string_search.hpp
    include/dynd/type_sequence.hpp
    include/dynd/type_promotion.hpp
    include/dynd/exceptions.hpp
    include/dynd/fpstatus.hpp
    include/dynd/functional.hpp
    include/dynd/json_formatter.hpp
    include/dynd/json_parser.hpp
    include/dynd/index.hpp
    include/dynd/irange.hpp
    include/dynd/option.hpp
    include/dynd/platform_definitions.hpp
    include/dynd/pointer.hpp
    include/dynd/shortvector.hpp
    include/dynd/string_encodings.hpp
    include/dynd/view.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/include/dynd/visibility.hpp
    include/dynd/with.hpp
    )

include_directories(
    include
    thirdparty/utf8/source
    )

source_group("Main Source" REGULAR_EXPRESSION "src/dynd/.*cpp")
source_group("Main Headers" REGULAR_EXPRESSION "include/dynd/.*hpp")
source_group("Types Source" REGULAR_EXPRESSION "src/dynd/types/.*cpp")
source_group("Types Headers" REGULAR_EXPRESSION "include/dynd/types/.*hpp")
source_group("Eval Source" REGULAR_EXPRESSION "src/dynd/eval/.*cpp")
source_group("Eval Headers" REGULAR_EXPRESSION "include/dynd/eval/.*hpp")
source_group("Func Source" REGULAR_EXPRESSION "src/dynd/func/.*cpp")
source_group("Func Headers" REGULAR_EXPRESSION "include/dynd/func/.*hpp")
source_group("Iter Source" REGULAR_EXPRESSION "src/dynd/iter/.*cpp")
source_group("Iter Headers" REGULAR_EXPRESSION "include/dynd/iter/.*hpp")
source_group("Kernels Source" REGULAR_EXPRESSION "src/dynd/kernels/.*cpp")
source_group("Kernels Headers" REGULAR_EXPRESSION "include/dynd/kernels/.*hpp")
source_group("MemBlock Source" REGULAR_EXPRESSION "src/dynd/memblock/.*cpp")
source_group("MemBlock Headers" REGULAR_EXPRESSION "include/dynd/memblock/.*hpp")
source_group("Internal Headers" REGULAR_EXPRESSION "src/dynd/.*hpp")

#set_source_files_properties(include/dynd/kernels/compare_kernels.hpp PROPERTIES COMPILE_FLAGS -Wno-sign-compare)
#set_source_files_properties(src/dynd/func/comparison.cpp PROPERTIES COMPILE_FLAGS -Wno-sign-compare)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

#if(DYND_LLVM)
#  if(LLVM_FOUND)
#    add_subdirectory(plugin)
#
#    add_dependencies(dynd_OBJ dynd_plugin)
#    set_target_properties(dynd_OBJ PROPERTIES COMPILE_FLAGS "-Xclang -load -Xclang plugin/libdynd_plugin.so")
#  endif()
#endif()

if (DYND_SHARED_LIB)
    # xcode doesn't like when all cpp files are contained in an OBJECT library
    add_library(libdyndt SHARED ${libdyndt_SRC})
    set_target_properties(libdyndt
        PROPERTIES
        OUTPUT_NAME "dyndt"
        PREFIX "lib"
        IMPORT_PREFIX "lib"
        )
    add_library(libdynd SHARED ${libdynd_SRC})
    set_target_properties(libdynd
        PROPERTIES
        OUTPUT_NAME "dynd"
        PREFIX "lib"
        IMPORT_PREFIX "lib"
        )
    add_dependencies(libdynd libdyndt)
else()
    add_library(libdyndt STATIC ${libdyndt_SRC})
    set_target_properties(libdyndt
        PROPERTIES
        OUTPUT_NAME "dyndt"
        PREFIX "lib"
        )

    add_library(libdynd STATIC ${libdynd_SRC})
    set_target_properties(libdynd
        PROPERTIES
        OUTPUT_NAME "dynd"
        PREFIX "lib"
        )
endif()

set_property(
    TARGET libdyndt
    PROPERTY COMPILE_DEFINITIONS DYNDT_EXPORT
)

set_property(
    TARGET libdynd
    PROPERTY COMPILE_DEFINITIONS DYND_EXPORT
)

# Add preprocessor definitions from CMake
configure_file("include/dynd/cmake_config.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/dynd/cmake_config.hpp")

if(APPLE)
    # The rpath stuff is confusing, and this is our attempt to get it right.
    # It's been complicated by a bug in certain versions of CMake that plays
    # particularly badly with CUDA. If anyone knows the "right" way to do this,
    # that would be great.
    #
    # See http://stackoverflow.com/questions/22885207/opencv-and-pcl-builds-on-osx-result-in-malformed-object-load-command-cmdsize
    #
    set_target_properties(libdyndt
        PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_NAME_DIR "@rpath"
    )
    set_target_properties(libdynd
        PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_NAME_DIR "@rpath"
    )
endif()

if (DYND_SHARED_LIB OR (NOT DYND_INSTALL_LIB))
    # If we're not making an installable static library,
    # link the sublibraries normally
    set(DYND_LINK_LIBS ${DYND_LINK_LIBS} libdyndt)
endif()

target_link_libraries(libdyndt ${DYNDT_LINK_LIBS})
target_link_libraries(libdynd ${DYND_LINK_LIBS})

if(DYND_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(DYND_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(DYND_BUILD_PLUGINS)
    add_subdirectory(plugins/mkl)
endif()

add_subdirectory(examples)

# Create a libdynd-config script
get_property(dynd_library_prefix TARGET libdynd PROPERTY PREFIX)
get_property(dynd_output_name TARGET libdynd PROPERTY OUTPUT_NAME)
get_property(dyndt_output_name TARGET libdyndt PROPERTY OUTPUT_NAME)
if (DYND_SHARED_LIB)
    if ("${CMAKE_IMPORT_LIBRARY_SUFFIX}" STREQUAL "")
        set(DYND_LIB_FILE "${dynd_library_prefix}${dynd_output_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(DYNDT_LIB_FILE "${dynd_library_prefix}${dyndt_output_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    else()
        set(DYND_LIB_FILE "${dynd_library_prefix}${dynd_output_name}${CMAKE_IMPORT_LIBRARY_SUFFIX}")
        set(DYNDT_LIB_FILE "${dynd_library_prefix}${dyndt_output_name}${CMAKE_IMPORT_LIBRARY_SUFFIX}")
    endif()
else()
    set(DYND_LIB_FILE "${dynd_library_prefix}${dynd_output_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(DYNDT_LIB_FILE "${dynd_library_prefix}${dyndt_output_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
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

if(DYND_BUILD_DOCS)
    find_package(Doxygen)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Building DyND documentation requires that Doxygen be installed.")
    endif()
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile" @ONLY)
    add_custom_target(dynd_documentation ALL
                      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile
                      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile
                      COMMENT "Building documentation for libdynd."
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      VERBATIM
                      )
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs DESTINATION docs)
endif()

# Install the libdyndt binary
install(TARGETS libdyndt DESTINATION lib${LIB_SUFFIX} COMPONENT ndt)

# Install the libdynd binary
install(TARGETS libdynd DESTINATION lib${LIB_SUFFIX} COMPONENT nd)

# Install the libdynd headers
install(DIRECTORY "include/dynd" DESTINATION "include" COMPONENT headers)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/dynd" DESTINATION "include" COMPONENT include)

# Install the libdynd-config script
if(WIN32)
  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/libdynd-config.bat" DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
  if(DYND_SHARED_LIB)
    install(FILES "$<TARGET_FILE_DIR:libdynd>/libdynd.dll" DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
    install(FILES "$<TARGET_FILE_DIR:libdyndt>/libdyndt.dll" DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
  endif()
  else()
    install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/libdynd-config" DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif()

set(CPACK_GENERATOR "TGZ")
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL nd ndt include)
set(CPACK_PACKAGE_VERSION "0.7.2")
include(CPack)
