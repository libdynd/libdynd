cmake_minimum_required(VERSION 2.6)
project(blosc)

set(CMAKE_VERBOSE_MAKEFILE 1)

set(ABC "testing!")

set(blosc_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/blosc/blosc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/blosc/blosclz.c
    ${CMAKE_CURRENT_SOURCE_DIR}/blosc/shuffle.c
    ${CMAKE_CURRENT_SOURCE_DIR}/blosc/blosc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/blosc/blosclz.h
    ${CMAKE_CURRENT_SOURCE_DIR}/blosc/shuffle.h
    )

if(WIN32)
    set(blosc_SRC
        ${blosc_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/blosc/win32/pthread.c
        ${CMAKE_CURRENT_SOURCE_DIR}/blosc/win32/pthread.h
        )
    include_directories(blosc/win32)
endif()

# Just supporting static library build for now (no .so/.dll yet)
add_library(blosc STATIC ${blosc_SRC})

if(UNIX)
    # On some 64-bit unix platforms it appears to be necessary to
    # explicitly say -fPIC to make static linking work
    if(${CMAKE_SIZEOF_VOID_P} MATCHES 8)
        set_target_properties(blosc PROPERTIES COMPILE_FLAGS "-fPIC")
    endif()
    target_link_libraries(blosc pthread)
endif()

add_subdirectory(bench)
add_subdirectory(tests)

