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
  set_property(SOURCE "simple_example.cpp" PROPERTY CUDA_SOURCE_FILE TRUE)
  cuda_add_executable(simple_example simple_example.cpp
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