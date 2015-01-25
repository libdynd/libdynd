#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(dynd_examples)

include_directories(
    ../include
  )

if (DYND_CUDA)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/simple_example.cu"
    COMMAND echo "\\\#include \\\"${CMAKE_CURRENT_SOURCE_DIR}/simple_example.cpp\\\"" > simple_example.cu
    )
  cuda_add_executable(simple_example ${CMAKE_CURRENT_BINARY_DIR}/simple_example.cu
    OPTIONS -ldynddev -L${CMAKE_BINARY_DIR}
  )
  set(LINK_LIBS ${LINK_LIBS} libdynddev)
else ()
  add_executable(simple_example
    simple_example.cpp
  )
  set(LINK_LIBS ${LINK_LIBS} libdynd)
endif ()

target_link_libraries(simple_example
    ${LINK_LIBS}
  )