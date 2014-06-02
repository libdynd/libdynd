cmake_minimum_required(VERSION 2.6)
project(cephes)

set(CMAKE_VERBOSE_MAKEFILE 1)

set(cephes_SRC
    airy.c
    cbrt.c
    const.c
    gamma.c
    j0.c
    j1.c
    jv.c
    polevl.c
    mtherr.c
    sf_error.c
    struve.c
    yn.c
    mconf.h
    )

# Just supporting static library build for now (no .so/.dll yet)
add_library(cephes STATIC ${cephes_SRC})

# On some 64-bit unix platforms it appears to be necessary to
# explicitly say -fPIC to make static linking work
if(UNIX AND ${CMAKE_SIZEOF_VOID_P} MATCHES 8)
    set_target_properties(cephes PROPERTIES COMPILE_FLAGS "-fPIC")
endif()
