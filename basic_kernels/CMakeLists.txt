#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(basic_kernels)

include_directories(
    )

set(basic_kernels_SRC
    elwise_kernels.cpp
    )

add_library(basic_kernels SHARED
    ${basic_kernels_SRC}
    )

target_link_libraries(basic_kernels
    )
