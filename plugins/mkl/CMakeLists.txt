#
# Copyright (C) 2011-16 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(libdynd-mkl)

include(FindBLAS)

set(CMAKE_MACOSX_RPATH 1)

message("${BLAS_LIBRARIES}")
message("${BLAS_LINKER_FLAGS}")
message("${BLAS_INCLUDE_DIR}")

include_directories(include)
include_directories(/opt/intel/mkl/include)
include_directories(/opt/intel/compilers_and_libraries_2016.3.210/linux/mkl/include)

#link_directories(/opt/intel/mkl/lib)

set(mkl_SRC
    src/dynd/mkl.cpp
)

add_library(dynd-mkl SHARED ${mkl_SRC})

target_link_libraries(dynd-mkl libdynd ${BLAS_LIBRARIES} pthread m dl)
