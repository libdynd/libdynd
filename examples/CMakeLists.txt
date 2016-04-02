#
# Copyright (C) 2011-16 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(dynd_examples)

include_directories(
    ../include
  )

add_executable(simple_example
  simple_example.cpp
)

target_link_libraries(simple_example libdynd)
