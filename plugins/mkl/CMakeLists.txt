#
# Copyright (C) 2011-16 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(libdynd-mkl)

set(CMAKE_MACOSX_RPATH 1)

include_directories(include)
include_directories(/opt/intel/mkl/include)

link_directories(/opt/intel/mkl/lib)

set(mkl_SRC
    src/dynd/mkl.cpp
)

add_library(dynd-mkl SHARED ${mkl_SRC})

target_link_libraries(dynd-mkl libdynd mkl_rt pthread m dl)
