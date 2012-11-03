cmake_minimum_required(VERSION 2.6)
project(blosc-tests)

set(CMAKE_VERBOSE_MAKEFILE 1)

add_executable(test_api test_api.cpp)
target_link_libraries(test_api
    blosc
    )

add_executable(test_basics test_basics.cpp)
target_link_libraries(test_basics
    blosc
    )
