cmake_minimum_required(VERSION 2.6)
project(datetime)

set(CMAKE_VERBOSE_MAKEFILE 1)

set(datetime_SRC
    src/datetime_main.cpp
    src/datetime_strings.cpp
    src/datetime_localtime.cpp
    src/datetime_busday.cpp
    include/datetime_types.h
    include/datetime_main.h
    include/datetime_strings.h
    include/datetime_localtime.h
    include/datetime_busday.h
    )

include_directories(include)

# Just supporting static library build for now (no .so/.dll yet)
add_library(datetime STATIC ${datetime_SRC})

# On some 64-bit unix platforms it appears to be necessary to
# explicitly say -fPIC to make static linking work
if(UNIX AND ${CMAKE_SIZEOF_VOID_P} MATCHES 8)
    set_target_properties(datetime PROPERTIES COMPILE_FLAGS "-fPIC")
endif()
