#
# Copyright (C) 2011-16 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(libdynd-mkl)

set(CMAKE_MACOSX_RPATH 1)

# There is no FindMKL, but this will find Intel's MKL
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

include_directories(include)

set(mkl_SRC
    src/dynd/mkl.cpp
)

add_library(dynd-mkl SHARED ${mkl_SRC})

target_link_libraries(dynd-mkl libdynd mkl_rt)

install(TARGETS dynd-mkl DESTINATION lib)

install(DIRECTORY "include/dynd" DESTINATION "include" COMPONENT headers)
