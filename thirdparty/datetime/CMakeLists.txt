cmake_minimum_required(VERSION 2.6)
project(datetime)

set(CMAKE_VERBOSE_MAKEFILE 1)

set(datetime_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/datetime_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/datetime_strings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/datetime_localtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/datetime_busday.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/datetime_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/datetime_main.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/datetime_strings.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/datetime_localtime.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/datetime_busday.h
    )

include_directories(include)

# Just supporting static library build for now (no .so/.dll yet)
add_library(datetime STATIC ${datetime_SRC})

# On some 64-bit unix platforms it appears to be necessary to
# explicitly say -fPIC to make static linking work
if(UNIX AND (${CMAKE_SIZEOF_VOID_P} MATCHES 8))
    set_target_properties(datetime PROPERTIES COMPILE_FLAGS "-fPIC")
endif()
