cmake_minimum_required(VERSION 2.6)
project(blosc_tests)

set(CMAKE_VERBOSE_MAKEFILE 1)

add_executable(test_blosc_api test_api.cpp)
target_link_libraries(test_blosc_api
    blosc
    )

add_executable(test_blosc_basics test_basics.cpp)
target_link_libraries(test_blosc_basics
    blosc
    )
