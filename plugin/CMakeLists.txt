#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.8.11)
project(plugin)

find_package(LLVM REQUIRED CONFIG)


add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_INSTALL_PREFIX}/lib /usr/lib/llvm-3.6/lib)

add_library(plugin SHARED
    plugin.cpp
)

  target_link_libraries(plugin 
    clangAST
    clangBasic
    clangFrontend
    LLVMSupport
    )
