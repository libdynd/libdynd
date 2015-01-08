#
# Copyright (C) 2011-14 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(dynd_examples)

include_directories(
    ../include
    )

if (DYND_CUDA)
cuda_add_executable(simple_example
    simple_example.cu
    )
else ()
add_executable(simple_example
    simple_example.cpp
    )
endif ()

target_link_libraries(simple_example
    libdynd
    )
