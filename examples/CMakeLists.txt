cmake_minimum_required(VERSION 2.8)
project(dnd_examples)

include_directories(
    ../include
    )

add_executable(simple_example
    simple_example.cpp
    )

target_link_libraries(simple_example
    dynamicndarray
    )
